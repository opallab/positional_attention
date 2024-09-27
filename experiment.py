import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import Transformer
from data import Dataset_Basic, DataLoader
from utils import append_positional_encoding, identity_pe, get_pe, get_loss

def run_experiment(target, data, device, experiment: str = 'sample_complexity', i: int = 0):
    """Run experiment with given parameters and return final losses for standard and positional transformer

    Args:
        target (str): target function to learn (one of 'sum', 'min', 'median', 'sort', 'minsum')
        data (dict): dictionary containing parameters for the experiment
        device (torch.device): device to run the experiment on
        experiment (str, optional): type of experiment (one of 'sample_complexity', 'scale_generalization`,
        `size`). Defaults to 'sample_complexity'.
        i (int, optional): If sample_complexity is True, this is the index of data['num_train_samples']. Otherwise
        it the index of data['low_test'] and data['high_test']. Defaults to 0.
    """
    n = data['n'][i] if experiment=='size' else data['n'][0]
    num_train_samples = data['num_train_samples'][i] if experiment=='sample_complexity' else data['num_train_samples'][0]
    num_test_samples = data['num_test_samples']
    num_additional_node = data['num_additional_node']
    lr = data['lr']
    batch_size = data['batch_size']
    shuffling = data['shuffling']
    low_train = data['low_train']
    high_train = data['high_train']
    low_test = data['low_test'][i] if experiment=='scale_generalization' else data['low_test'][0]
    high_test = data['high_test'][i] if experiment=='scale_generalization' else data['high_test'][0]
    cumulative = data['cumulative']
    use_integer = data['use_integer']
    variable_length = data['variable_length'] if 'variable_length' in data else False

    if target == 'minsum':
        pos_enc_base = identity_pe(2*n+num_additional_node).to(device)
    else:
        pos_enc_base = identity_pe(n+num_additional_node).to(device)

    if target == 'path':
        data_dim = n
    else:
        data_dim = 1

    in_dim_s = data_dim + pos_enc_base.size(1)
    in_dim_p = data_dim
    out_dim = data_dim
    embed_dim = data['embed_dim']
    num_heads = data['num_heads']
    use_rope = data['RoPE'] if 'RoPE' in data else False
    num_layers = np.log2(n).astype(int) + 1 if 'model_num_layers' not in data else data['model_num_layers']
    mlp_hidden_dim = data['mlp_hidden_dim']
    mlp_num_layers = data['mlp_num_layers']

    epochs = data['epochs']
    final_losses_s = []
    final_losses_p = []
    for run in range(data['runs']):
        print(f"Run {run+1} / {data['runs']}:")
        train_dataset = Dataset_Basic(num_samples=num_train_samples, length=n, low=low_train, high=high_train, target=target, use_integer=use_integer, cumulative=cumulative, num_additional_node=num_additional_node, variable_length=variable_length)
        val_dataset =  Dataset_Basic(num_samples=num_test_samples, length=n, low=low_train, high=high_train, target=target, use_integer=use_integer, cumulative=cumulative, num_additional_node=num_additional_node, variable_length=variable_length)
        test_dataset = Dataset_Basic(num_samples=num_test_samples, length=n, low=low_test, high=high_test, target=target, use_integer=use_integer, cumulative=cumulative, num_additional_node=num_additional_node, reject_low=low_train, reject_high=high_train, variable_length=variable_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=shuffling, variable_length=variable_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, variable_length=variable_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, variable_length=variable_length)

        model_s = Transformer(in_dim=in_dim_s, embed_dim=embed_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers,
                        mlp_hidden_dim=mlp_hidden_dim, mlp_num_layers=mlp_num_layers, positional=False, RoPE=use_rope, pos_dim=pos_enc_base.size(1)).to(device)
        model_p = Transformer(in_dim=in_dim_p, embed_dim=embed_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers,
                            mlp_hidden_dim=mlp_hidden_dim, mlp_num_layers=mlp_num_layers, positional=True, pos_dim=pos_enc_base.size(1)).to(device)
        optimizer_s = torch.optim.Adam(model_s.parameters(), lr=lr, weight_decay=data["weight_decay"])
        optimizer_p = torch.optim.Adam(model_p.parameters(), lr=lr, weight_decay=data["weight_decay"])
        scheduler_s = ReduceLROnPlateau(optimizer_s, mode='min', patience=50, factor=0.9, min_lr=1.0e-6)
        scheduler_p = ReduceLROnPlateau(optimizer_p, mode='min', patience=50, factor=0.9, min_lr=1.0e-6)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model_s.train()
            model_p.train()
            train_loss_s = 0
            train_loss_p = 0
            loss_s = 0
            loss_p = 0

            for (x, y) in train_loader:
                x, y = x.to(device), y.to(device)
                pos_enc = get_pe(pos_enc_base, x, num_additional_node) if variable_length else pos_enc_base
                x_app = append_positional_encoding(x, pos_enc)

                optimizer_s.zero_grad()
                out = model_s(x_app, p=pos_enc)
                loss_s = get_loss(criterion, out, y, num_additional_node, n, target)
                loss_s.backward()
                optimizer_s.step()
                train_loss_s += loss_s.item()

                optimizer_p.zero_grad()
                out = model_p(x, p=pos_enc)
                loss_p = get_loss(criterion, out, y, num_additional_node, n, target)
                loss_p.backward()
                optimizer_p.step()
                train_loss_p += loss_p.item()

            scheduler_s.step(train_loss_s)
            scheduler_p.step(train_loss_p)

            if epoch % 10 == 0:
                with torch.no_grad():
                    val_loss_s, test_loss_s = 0, 0
                    val_loss_p, test_loss_p = 0, 0
                    for (x, y) in val_loader:
                        x, y = x.to(device), y.to(device)
                        pos_enc = get_pe(pos_enc_base, x, num_additional_node) if variable_length else pos_enc_base
                        x_app = append_positional_encoding(x, pos_enc)
                        out = model_s(x_app, p=pos_enc)
                        val_loss_s += get_loss(criterion, out, y, num_additional_node, n, target).item()
                        out = model_p(x, p=pos_enc)
                        val_loss_p += get_loss(criterion, out, y, num_additional_node, n, target).item()

                    for (x, y) in test_loader:
                        x, y = x.to(device), y.to(device)
                        pos_enc = get_pe(pos_enc_base, x, num_additional_node) if variable_length else pos_enc_base
                        x_app = append_positional_encoding(x, pos_enc)
                        out = model_s(x_app, p=pos_enc)
                        test_loss_s += get_loss(criterion, out, y, num_additional_node, n, target).item()
                        out = model_p(x, p=pos_enc)
                        test_loss_p += get_loss(criterion, out, y, num_additional_node, n, target).item()

                    train_loss_s /= len(train_loader)
                    train_loss_p /= len(train_loader)
                    val_loss_s /= len(val_loader)
                    val_loss_p /= len(val_loader)
                    print(f"Epoch {epoch}, standard train/val/test: {train_loss_s:.4e}/{val_loss_s:.4e}/{test_loss_s}, positional train/val/test: {train_loss_p:.4e}/{val_loss_p:.4e}/{test_loss_p:.4e}")
                    if epoch == epochs-1:
                        final_losses_s.append((train_loss_s, val_loss_s, test_loss_s))
                        final_losses_p.append((train_loss_p, val_loss_p, test_loss_p))

            if epoch % 100 == 0:
                print("Learning rate for standard transformer: ", optimizer_s.param_groups[0]['lr'])
                print("Learning rate for positional transformer: ", optimizer_p.param_groups[0]['lr'])

    return final_losses_s, final_losses_p


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--savepath', type=str, required=True)
    argparser.add_argument('--params', type=str, required=True)
    argparser.add_argument('--task', type=str, required=True)
    args = argparser.parse_args()

    print("PyTorch version:", torch.__version__)
    print("Access to GPU:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.params, 'r') as fp:
        data = json.load(fp)

    os.makedirs(args.savepath, exist_ok=True)
    with open(args.savepath + "/params.json", 'w') as fp:
        json.dump(data, fp)

    experiment = 'size'
    times = len(data['n'])
    if len(data['num_train_samples']) > 1:
        experiment = 'sample_complexity'
        times = len(data['num_train_samples'])
    if len(data['low_test']) > 1:
        assert len(data['low_test']) == len(data['high_test']), "Length of low_test and high_test should be the same"
        experiment = 'scale_generalization'
        times = len(data['low_test'])

    variable_length = data['variable_length'] if 'variable_length' in data else False
    use_rope = data['RoPE'] if 'RoPE' in data else False

    print(f"Experiment: {experiment}")
    print(f"Task: {args.task}")
    print(f"Variable length: {variable_length}")
    print(f"Using RoPE: {use_rope}")
    for i in range(times):
        n = data['n'][i] if experiment=='size' else data['n'][0]
        num_train_samples = data['num_train_samples'][i] if experiment=='sample_complexity' else data['num_train_samples'][0]
        low_test = data['low_test'][i] if experiment=='scale_generalization' else data['low_test'][0]
        high_test = data['high_test'][i] if experiment=='scale_generalization' else data['high_test'][0]

        print(f"n: {n}, Training samples: {num_train_samples}, Test range: [{low_test}, {high_test}]")
        final_losses_s, final_losses_p = run_experiment(args.task, data, device, experiment=experiment, i=i)
        print("===============================================")

        filename = f"/train_val_test_n{n}_l{low_test}_h{high_test}_samples{num_train_samples}"
        filename_s = args.savepath + filename + "_standard.txt"
        filename_p = args.savepath + filename + "_positional.txt"

        os.makedirs(os.path.dirname(filename_s), exist_ok=True)
        os.makedirs(os.path.dirname(filename_p), exist_ok=True)

        with open(filename_s, 'w') as f:
            for item in final_losses_s:
                print(f"{item[0]}\t{item[1]}\t{item[2]}", file=f)

        with open(filename_p, 'w') as f:
            for item in final_losses_p:
                print(f"{item[0]}\t{item[1]}\t{item[2]}", file=f)
