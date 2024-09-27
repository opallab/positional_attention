import random
import warnings
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler, DataLoader

def max_subarray(x):
    best_sum = float('-inf')
    current_sum = 0
    for a in x:
        current_sum = max(a, current_sum + a)
        best_sum = max(best_sum, current_sum)
    return best_sum

def reverse_cumsum(x):
    return x +  x.sum() - torch.cumsum(x, dim=0)

def generate_data_basic(length, low, high, target='sum', use_integer=False, cumulative=False, num_additional_node=0, reject_low=None, reject_high=None):
    """
    Input
        length: input length
        low: lower bound of input number range
        high: upper bound of input number range
        use_integer: input numbers are sampled from integers
        cumulative: generate cumulative version of the problem
        num_additional_node: number of additinoal nodes added to the input, "length of the scratchpad"
    """
    if reject_low == low and reject_high == high:
        reject_low, reject_high = None, None
        
    if target == 'minsum':
        length = 2*length
        
    if use_integer:
        X = torch.randint(low=low, high=high, size=(length,)).float()
    else:
        if reject_low is None or reject_high is None:
            ranges = (high - low)*torch.rand(2) + low
            low, high = ranges.min(), ranges.max()
        else:
            low_i, high_i = 0, 0
            while True:
                low_i = low + (high - low)*torch.rand(1)
                high_i = low + (high - low)*torch.rand(1)
                if low_i > high_i:
                    low_i, high_i = high_i, low_i
                if not (low_i >= reject_low and high_i <= reject_high):
                    break
            low, high = low_i, high_i
        X = low + (high - low)*torch.rand(length)

    if target == 'sum':
        Y = torch.cumsum(X, dim=0) if cumulative else torch.full((length,), X.sum())
    elif target == 'min':
        Y = torch.cummin(X, dim=0)[0] if cumulative else torch.full((length,), X.min())
    elif target == 'max':
        Y = torch.cummax(X, dim=0)[0] if cumulative else torch.full((length,), X.max())
    elif target == 'median':
        if cumulative:
            Y = torch.tensor([torch.quantile(X[:i+1], 0.5, interpolation='midpoint') for i in range(len(X))])
        else:
            Y = torch.full((length,), torch.quantile(X, 0.5, interpolation='midpoint'))
    elif target == 'sort':
        if cumulative:
            warnings.warn("Cumulative version of sort is not defined, setting cumultive = False")
        Y = torch.sort(X, dim=0)[0]
    elif target == 'minsum':
        Y = X[:length//2] + X[length//2:]
        Y = torch.cummin(Y, dim=0)[0] if cumulative else torch.full((length//2,), Y.min())
        Y = torch.cat([Y, Y], dim=-1)
    elif target == 'maxsub':
        if cumulative:
            Y = torch.tensor([max_subarray(X[:i+1]) for i in range(len(X))])
        else:
            Y = torch.full((length,), max_subarray(X))
  
    if num_additional_node > 0:
        padding = torch.zeros(num_additional_node)
        X = torch.cat([X, padding], dim=0)
        Y = torch.cat([Y, padding], dim=0)
    return X.unsqueeze(-1), Y.unsqueeze(-1)

def generate_shortest_path_data(n, low, high, use_integer=False, num_additional_node=0, reject_low=None, reject_high=None):
    if reject_low == low and reject_high == high:
        reject_low, reject_high = None, None

    if use_integer:
        edge_weights = torch.randint(low=low, high=high, size=(n-1,)).float()
    else:
        if reject_low is None or reject_high is None:
            ranges = (high - low)*torch.rand(2) + low
            low, high = ranges.min(), ranges.max()
        else:
            low_i, high_i = 0, 0
            while True:
                low_i = low + (high - low)*torch.rand(1)
                high_i = low + (high - low)*torch.rand(1)
                if low_i > high_i:
                    low_i, high_i = high_i, low_i
                if not (low_i >= reject_low and high_i <= reject_high):
                    break
            low, high = low_i, high_i
        edge_weights = low + (high - low)*torch.rand(n-1)
        
    perm = torch.randperm(n)
    X = torch.diag(edge_weights, 1) + torch.diag(edge_weights, -1)
    Y = torch.zeros(n,n)
    for i in range(n):
        Y[i,:i] = reverse_cumsum(edge_weights[:i])
        Y[i,(i+1):] = torch.cumsum(edge_weights[i:], dim=0)
    X = X[:,perm]
    X = X[perm,:]
    Y = Y[:,perm]
    Y = Y[perm,:]
    if num_additional_node > 0:
        padding = torch.zeros(num_additional_node, n)
        X = torch.cat([X, padding], dim=0)
        Y = torch.cat([Y, padding], dim=0)
    return X, Y


class Dataset_Basic(Dataset):
    def __init__(self, num_samples, length, low, high, target='sum', use_integer=False, cumulative=False, num_additional_node=0, reject_low=None, reject_high=None, variable_length=True, length_low=1):
        self.dataset = []
        self.variable_length = variable_length
        self.num_samples = num_samples
        if variable_length:
            self.samples_per_length = num_samples // (length-length_low+1)
            self.length = length

            self.idx_map = []
            self.length_to_indices = defaultdict(list)

            idx = 0
            samples = self.samples_per_length
            for len_i in range(length_low, length+1):
                current_length_samples = []
                if len_i == length:
                    samples += num_samples % length
                for _ in range(samples):
                    data = generate_data_basic(
                        length=len_i,
                        low=low,
                        high=high,
                        target=target,
                        use_integer=use_integer,
                        cumulative=cumulative,
                        num_additional_node=num_additional_node,
                        reject_low=reject_low,
                        reject_high=reject_high
                    )
                    current_length_samples.append(data)
                    self.idx_map.append((len_i - length_low, len(current_length_samples) - 1))
                    self.length_to_indices[len_i].append(idx)
                    idx += 1
                self.dataset.append(current_length_samples)
        else:
            if target == 'path':
                for _ in range(num_samples):
                    self.dataset.append(generate_shortest_path_data(n=length, low=low, high=high, use_integer=use_integer, num_additional_node=num_additional_node, reject_low=reject_low, reject_high=reject_high))
            else:
                for _ in range(num_samples):
                    self.dataset.append(generate_data_basic(length=length,low=low, high=high, target=target, use_integer=use_integer, cumulative=cumulative, num_additional_node=num_additional_node, reject_low=reject_low, reject_high=reject_high))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.variable_length:
            i, j = self.idx_map[idx]
            return self.dataset[i][j]
        else:
            return self.dataset[idx]

class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.length_to_indices = dataset.length_to_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = self.create_batches() # precompute batches

    def create_batches(self):
        batches = []
        for length, indices in self.length_to_indices.items():
            indices = indices.copy()
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        if self.shuffle:
            self.batches = self.create_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

class DataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, variable_length=True, **kwargs):
        if variable_length:
            sampler = CustomBatchSampler(dataset, batch_size, shuffle)
            super().__init__(dataset, batch_sampler=sampler, **kwargs)
        else:
            super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)