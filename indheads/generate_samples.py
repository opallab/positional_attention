import argparse
import os
import random

import torch
from mingpt.utils import set_seed, setup_logging
from data import IndDataset, InductionHopsFullSequenceTask, get_config
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a GPT on induction heads task')
    parser.add_argument('--work_dir', type=str, help='output directory')
    parser.add_argument('--num_samples', type=int, default=50_000, help='number of samples to generate')
    parser.add_argument('--seq_len', type=int, default=100, help='number of characters in sequence')
    parser.add_argument('--min_hops', type=int, default=0, help='minimum number of hops')
    parser.add_argument('--max_hops', type=int, default=3, help='maximum number of hops')
    parser.add_argument('--trials', type=int, default=10, help='number of trials')

    config = get_config()
    args = parser.parse_args()

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

        generator = InductionHopsFullSequenceTask(seq_len=args.seq_len, char_tokens=10, min_hops=args.min_hops, max_hops=args.max_hops, rng=config.system.seed, offset=-5)
        train_dataset = IndDataset(config.data, generator, args.num_samples)
        val_dataset = IndDataset(config.data, generator, 1000)
        
        generator = InductionHopsFullSequenceTask(seq_len=args.seq_len, char_tokens=10, min_hops=args.min_hops, max_hops=args.max_hops, rng=config.system.seed, offset=5)
        test_dataset = IndDataset(config.data, generator, 1000)
        torch.save(train_dataset, f'{args.work_dir}/train_dataset_{t}.pt')
        torch.save(val_dataset, f'{args.work_dir}/val_dataset_{t}.pt')
        torch.save(test_dataset, f'{args.work_dir}/test_dataset_{t}.pt')