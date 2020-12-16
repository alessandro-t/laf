import numpy as np
import os
import torch

from utils import get_labels, np_to_torch, torch_to_np 


class Dataset:
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)[()]
        self.X = data['X']
        self.y = data['y']
       
    def get_minibatches(self, batch_size=64, shuffle=True, drop_reminder=True, use_torch=False, device='cpu'):
        idx = np.arange(len(self.X))
        if shuffle:
            np.random.shuffle(idx)
        if drop_reminder:
            n_batches = len(idx) // batch_size
        else:
            n_batches = np.ceil(len(idx) / batch_size).astype(np.int32)
        for b in range(n_batches):
            li = b*batch_size
            ri = min((b+1)*batch_size, len(idx))
            current_idx = idx[li:ri]
            xbatch = np.concatenate(self.X[current_idx])
            ybatch = self.y[current_idx]
            sbatch = np.concatenate([[s]*len(self.X[c]) for s,c in enumerate(current_idx)])
            
            if use_torch:
                 yield np_to_torch(xbatch, dtype=torch.long, device=device), \
                       np_to_torch(sbatch, dtype=torch.long, device=device), \
                       np_to_torch(ybatch, dtype=torch.float32, device=device)
            else:
                 yield xbatch, sbatch, ybatch


