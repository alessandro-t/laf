import torch

def get_labels():
    LABELS = ['max', 'min', 'count', 'inv_count', 'sum', \
              'mean', 'median', 'min_over_max', \
              'max_minus_min', 'std', 'skew', \
              'kurtosis']
    #LABELS = ['max', 'min', 'count', 'inv_count', 'sum', \
    #          'mean', 'median', 'min_over_max', \
    #          'median_sum', 'quantile', \
    #          'max_minus_min', 'std', 'skew', \
    #          'kurtosis']
    return LABELS

def np_to_torch(x, dtype, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)

def torch_to_np(x):
    return x.detach().cpu().numpy()
