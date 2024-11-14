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

def run_experiment(target, data, device, model_savepath=None, run_id=0):
    """Run experiment with given parameters and return final losses for standard and positional transformer

    Args:
        target (str): target function (one of 'sum', 'min', 'median', 'sort', 'minsum')
        data (dict): dictionary containing parameters for the experiment
        device (torch.device): device to run the experiment on
    """
    n = data['n'][0]
    num_train_samples = data['num_train_samples'][0]
    num_test_samples = data['num_test_samples']
    num_additional_node = data['num_additional_node']
    lr = data['lr']
    batch_size = data['batch_size']
    shuffling = data['shuffling']
    low_train = data['low_train']
    high_train = data['high_train']
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
    num_layers = np.log2(n).astype(int) + 1 if 'model_num_layers' not in data else data['model_num_layers']
    mlp_hidden_dim = data['mlp_hidden_dim']
    mlp_num_layers = data['mlp_num_layers']

    epochs = data['epochs']

    train_dataset = Dataset_Basic(num_samples=num_train_samples, length=n, low=low_train, high=high_train, target=target, use_integer=use_integer, cumulative=cumulative, num_additional_node=num_additional_node, variable_length=variable_length)
    val_dataset =  Dataset_Basic(num_samples=num_test_samples, length=n, low=low_train, high=high_train, target=target, use_integer=use_integer, cumulative=cumulative, num_additional_node=num_additional_node, variable_length=variable_length)
    test_datasets = [Dataset_Basic(num_samples=num_test_samples, length=n, low=low_test, high=high_test, target=target, use_integer=use_integer, cumulative=cumulative, num_additional_node=num_additional_node, reject_low=low_train, reject_high=high_train, variable_length=variable_length) for low_test, high_test in zip(data['low_test'], data['high_test'])]

    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=shuffling, variable_length=variable_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, variable_length=variable_length)
    test_loaders = [DataLoader(test_dataset, batch_size=batch_size, shuffle=False, variable_length=variable_length) for test_dataset in test_datasets]

    # Standard, Input X|P, Attention X
    model_0 = Transformer(in_dim=in_dim_s, embed_dim=embed_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers,
                          mlp_hidden_dim=mlp_hidden_dim, mlp_num_layers=mlp_num_layers, positional=False, hybrid=False, RoPE=False,
                          pos_dim=pos_enc_base.size(1)).to(device)
    # Positional, Input X|P, Attention P
    model_1 = Transformer(in_dim=in_dim_s, embed_dim=embed_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers,
                          mlp_hidden_dim=mlp_hidden_dim, mlp_num_layers=mlp_num_layers, positional=True, hybrid=False, RoPE=False,
                          pos_dim=pos_enc_base.size(1)).to(device)
    # Standard, Input X, Attention X
    model_2 = Transformer(in_dim=in_dim_p, embed_dim=embed_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers,
                          mlp_hidden_dim=mlp_hidden_dim, mlp_num_layers=mlp_num_layers, positional=False, hybrid=False, RoPE=False,
                          pos_dim=pos_enc_base.size(1)).to(device)
    # Standard, Input X, Attention X|P
    model_3 = Transformer(in_dim=in_dim_p, embed_dim=embed_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers,
                          mlp_hidden_dim=mlp_hidden_dim, mlp_num_layers=mlp_num_layers, positional=False, hybrid=True, RoPE=False,
                          pos_dim=pos_enc_base.size(1)).to(device)
    # Standard, Input X, Attention X + RoPE
    model_4 = Transformer(in_dim=in_dim_p, embed_dim=embed_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers,
                          mlp_hidden_dim=mlp_hidden_dim, mlp_num_layers=mlp_num_layers, positional=False, hybrid=False, RoPE=True,
                          pos_dim=pos_enc_base.size(1)).to(device)
    # Positional, Input X, Attention P
    model_5 = Transformer(in_dim=in_dim_p, embed_dim=embed_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers,
                          mlp_hidden_dim=mlp_hidden_dim, mlp_num_layers=mlp_num_layers, positional=True, hybrid=False, RoPE=False,
                          pos_dim=pos_enc_base.size(1)).to(device)
    models = [model_0, model_1, model_2, model_3, model_4, model_5]
    optimizers = [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=data["weight_decay"]) for model in models]
    schedulers = [ReduceLROnPlateau(opt, mode='min', patience=50, factor=0.9, min_lr=1.0e-6) for opt in optimizers]
    criterion = nn.MSELoss()

    for epoch in range(epochs):

        for model in models:
            model.train()
        train_loss = [0]*len(models)
        loss = [0]*len(models)

        for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)
            pos_enc = get_pe(pos_enc_base, x, num_additional_node) if variable_length else pos_enc_base
            x_app = append_positional_encoding(x, pos_enc)

            for i in range(0,2):
                optimizers[i].zero_grad()
                out = models[i](x_app, p=pos_enc)
                loss[i] = get_loss(criterion, out, y, num_additional_node, n, target)
                loss[i].backward()
                optimizers[i].step()
                train_loss[i] += loss[i].item()

            for i in range(2,6):
                optimizers[i].zero_grad()
                out = models[i](x, p=pos_enc)
                loss[i] = get_loss(criterion, out, y, num_additional_node, n, target)
                loss[i].backward()
                optimizers[i].step()
                train_loss[i] += loss[i].item()

        for scheduler, l in zip(schedulers,train_loss):
            scheduler.step(l)

        if epoch % 10 == 0:
            with torch.no_grad():
                val_loss = [0]*len(models)
                test_loss = [[0]*len(test_loaders) for _ in range(len(models))]
                for (x, y) in val_loader:
                    x, y = x.to(device), y.to(device)
                    pos_enc = get_pe(pos_enc_base, x, num_additional_node) if variable_length else pos_enc_base
                    x_app = append_positional_encoding(x, pos_enc)
                    for i in range(0,2):
                        out = models[i](x_app, p=pos_enc)
                        val_loss[i] += get_loss(criterion, out, y, num_additional_node, n, target).item()
                    for i in range(2,6):
                        out = models[i](x, p=pos_enc)
                        val_loss[i] += get_loss(criterion, out, y, num_additional_node, n, target).item()

                for j, test_loader in enumerate(test_loaders):
                    for (x, y) in test_loader:
                        x, y = x.to(device), y.to(device)
                        pos_enc = get_pe(pos_enc_base, x, num_additional_node) if variable_length else pos_enc_base
                        x_app = append_positional_encoding(x, pos_enc)
                        for i in range(0,2):
                            out = models[i](x_app, p=pos_enc)
                            test_loss[i][j] += get_loss(criterion, out, y, num_additional_node, n, target).item()
                        for i in range(2,6):
                            out = models[i](x, p=pos_enc)
                            test_loss[i][j] += get_loss(criterion, out, y, num_additional_node, n, target).item()

                train_loss = [l/len(train_loader) for l in train_loss]
                val_loss = [l/len(val_loader) for l in val_loss]
                print(f"Epoch {epoch}")
                for i in range(len(models)):
                    print(f"M{i} train/val: {train_loss[i]:.4e}/{val_loss[i]:.4e}")
                    str_loss = "/".join([f"{loss:.4e}" for loss in test_loss[i]])
                    print(f"Test: {str_loss}")

                if epoch == epochs-1:
                    final_losses = [[train_loss[i]] + [val_loss[i]] + test_loss[i] for i in range(len(models))]

        if epoch % 100 == 0:
            for i, opt in enumerate(optimizers):
                print(f"Learning rate for M{i}: ", opt.param_groups[0]['lr'])

    for i,model in enumerate(models):
        torch.save(model.state_dict(), model_savepath + f"/M{i}_run{run_id}.pt")

    return final_losses


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

    model_savepath = args.savepath + '/models'
    os.makedirs(model_savepath, exist_ok=True)

    assert len(data['n']) == 1, "Only one value of n should be provided"
    assert len(data['num_train_samples']) == 1, "Only one value of num_train_samples should be provided"
    assert len(data['low_test']) == len(data['high_test']), "Length of low_test and high_test should be the same"
    experiment = 'scale_generalization'
    times = len(data['low_test'])
    variable_length = data['variable_length'] if 'variable_length' in data else False

    print(f"Experiment: {experiment}")
    print(f"Task: {args.task}")
    print(f"Variable length: {variable_length}")
    n = data['n'][0]
    num_train_samples = data['num_train_samples'][0]
    low_test = data['low_test'][0] if experiment=='scale_generalization' else data['low_test'][0]
    high_test = data['high_test'][0] if experiment=='scale_generalization' else data['high_test'][0]

    filename = args.savepath + f"/train_val_test_scale_n{n}_samples{num_train_samples}"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print(f"n: {n}, Training samples: {num_train_samples}, Test range: [{low_test}, {high_test}]")
    for run in range(data['runs']):
        print(f"Run {run+1} / {data['runs']}:")
        final_losses = run_experiment(args.task, data, device, model_savepath=model_savepath, run_id=run)

        for i in range(len(final_losses)):
            with open(filename + f"_M{i}", 'a') as f:
                for loss in final_losses[i]:
                    print(f"{loss:.10e}\t", end='', file=f)
                print("", file=f)

    print("===============================================")
