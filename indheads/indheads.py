"""
Trains a character-level language model.
"""
import argparse
import os
import random

import torch
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging
from data import IndDataset, get_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on the k-hop induction heads task')
    parser.add_argument('--work_dir', type=str, default=None, help='output directory')
    parser.add_argument('--data_dir', type=str, default=None, help='data directory')
    parser.add_argument('--iters', type=int, default=int(1e5), help='number of iteration steps to train for')
    parser.add_argument('--num_samples', type=int, default=-1, help='number of samples')
    parser.add_argument('--positional', action='store_true', help='use positional attention')
    parser.add_argument('--freq', type=int, default=5000, help='frequency of logging')
    parser.add_argument('--layers', type=int, default=3, help='number of layers')
    parser.add_argument('--trials', type=int, default=10, help='number of trials')
    parser.add_argument('--no_causal', action='store_true', help='disable causal attention')
    parser.set_defaults(positional=False)
    parser.set_defaults(no_causal=False)

    config = get_config()
    args = parser.parse_args()

    config.model.positional = args.positional
    config.model.no_causal = args.no_causal
    model_type = "positional" if args.positional else "standard"
    config.trainer.max_iters = args.iters
    if args.work_dir is not None:
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)
        config.system.work_dir = args.work_dir

    random.seed(config.system.seed)
    seeds = [random.randint(0, 9999) for _ in range(args.trials)]

    for t in range(args.trials):
    
        config.system.seed = seeds[t]
        setup_logging(config)
        set_seed(config.system.seed)

        train_dataset = torch.load(os.path.join(args.data_dir, f'train_dataset_{t}.pt'), weights_only=False)
        if args.num_samples > 0:
            X, y = train_dataset[:args.num_samples]
            train_dataset.X = X
            train_dataset.y = y
        val_dataset = torch.load(os.path.join(args.data_dir, f'val_dataset_{t}.pt'), weights_only=False)
        test_dataset = torch.load(os.path.join(args.data_dir, f'test_dataset_{t}.pt'), weights_only=False)

        if t > 0:
            config.model.model_type = None
        else:
            config.model.model_type = f'gpt-nano-{args.layers}'

        config.model.vocab_size = 11 + 5 # 5 are OOD tokens
        config.model.block_size = train_dataset[0][0].shape[0]

        model = GPT(config.model)
        trainer = Trainer(config.trainer, model, train_dataset)

        def batch_end_callback(trainer):
            if (trainer.iter_num+1) % args.freq == 0 or trainer.iter_num == args.iters-1:

                train_loss = trainer.cum_loss / max((trainer.iter_num - trainer.last_iter_num), 1)
                trainer.last_iter_num = trainer.iter_num
                trainer.cum_loss = 0.0
                model.eval()
                with torch.no_grad():
                    X, y = val_dataset[:]
                    X, y = X.to(trainer.device), y.to(trainer.device)
                    _, eval_loss = model(X, y)

                    X, y = test_dataset[:]
                    X, y = X.to(trainer.device), y.to(trainer.device)
                    _, test_loss = model(X, y)

                    print(f"iter {trainer.iter_num}, train loss: {train_loss:.5f}, eval loss: {eval_loss:.5f}, test loss: {test_loss:.5f}")
                    trainer.log_train_loss = train_loss
                    trainer.log_eval_loss = eval_loss
                    trainer.log_test_loss = test_loss

                ckpt_path = os.path.join(config.system.work_dir, f"model_trial_{t}_{model_type}.pt")
                torch.save(model.state_dict(), ckpt_path)
                model.train()

        trainer.set_callback('on_batch_end', batch_end_callback)
        trainer.run()
        
        filename = os.path.join(config.system.work_dir, f'{model_type}.txt')
        with open(filename, 'a') as f:
                for loss in [trainer.log_train_loss, trainer.log_eval_loss, trainer.log_test_loss]:
                    print(f"{loss:.10e}\t", end='', file=f)
                print("", file=f)
