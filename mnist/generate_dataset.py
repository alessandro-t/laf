from torchvision import datasets
from tqdm.auto import tqdm
import argparse
import numpy as np
import os
import torch

from utils import get_labels, np_to_torch, torch_to_np 

def generate_dataset(dataset_idx, instance_labels, n_sets, setsize, label_type, bias):
    sets = []
    labels = []
    if bias:
        class_dict = {i:np.where(instance_labels == i)[0] for i in range(10)}
        if label_type == 'max':
            counts = np.zeros(10, dtype=np.int32)
            for s in range(n_sets):
                idx = s%10
                labels.append(idx)
                elements = np.random.randint(1, setsize + 1)
                counts *= 0
                counts[idx] = 1
                classes = np.random.choice(range(0,idx+1), elements-1)
                counts += np.bincount(classes,minlength=10)
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
        elif label_type == 'min':
            counts = np.zeros(10, dtype=np.int32)
            for s in range(n_sets):
                idx = s%10
                labels.append(idx)
                elements = np.random.randint(1, setsize + 1)
                counts *= 0
                counts[idx] = 1
                classes = np.random.choice(range(idx,10), elements-1)
                counts += np.bincount(classes, minlength=10)
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
        elif label_type == 'count':
            for s in range(n_sets):
                elements = np.random.randint(1, setsize + 1) 
                elements = np.random.beta(0.5,0.15,size=1)*setsize
                elements = np.maximum(np.round(elements).astype(np.int32), 1)
                idxs = np.random.choice(dataset_idx, elements)
                sets.append(idxs)
                labels.append(elements)
        elif label_type == 'inv_count':
            for s in range(n_sets):
                elements = np.random.randint(1, setsize + 1) 
                elements = np.random.beta(0.15,0.5,size=1)*setsize
                elements = np.maximum(np.round(elements).astype(np.int32), 1)
                idxs = np.random.choice(dataset_idx, elements)
                sets.append(idxs)
                labels.append(1./elements)
        elif label_type == 'sum':
            lower = np.round(np.random.beta(0.1 + 0.1 * setsize/50, 0.37,size=(n_sets//3, \
                                            setsize))*9).astype(np.int32)
            upper = np.round(np.random.beta(0.37, 0.1 + 0.1 * setsize/50,size=(n_sets//3, \
                                            setsize))*9).astype(np.int32)
            middle = np.random.randint(0, 10, size=(n_sets-2*(n_sets//3), setsize))
            
            samples = np.concatenate((lower, middle, upper), axis=0)
            all_counts = np.array(list(map(lambda t: np.bincount(t, minlength=10), samples)))
            labels = list(np.sum(samples, axis=1))
            for counts in all_counts:
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
        elif label_type == 'mean':
            lower = np.round(np.random.beta(0.1 + 0.1 * setsize/50, 0.37,size=(n_sets//3, \
                                            setsize))*9).astype(np.int32)
            upper = np.round(np.random.beta(0.37, 0.1 + 0.1 * setsize/50,size=(n_sets//3, \
                                            setsize))*9).astype(np.int32)
            middle = np.random.randint(0, 10, size=(n_sets-2*(n_sets//3), setsize))
            
            samples = np.concatenate((lower, middle, upper), axis=0)
            all_counts = np.array(list(map(lambda t: np.bincount(t, minlength=10), samples)))
            labels = list(np.mean(samples.astype(np.float32), axis=1))
            for counts in all_counts:
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
        elif label_type == 'median':
            lower = np.round(np.random.beta(0.2 ,0.3,size=(int(42*n_sets/100), \
                                            setsize))*9).astype(np.int32)
            upper = np.round(np.random.beta(0.3, 0.2,size=(int(42*n_sets/100), \
                                            setsize))*9).astype(np.int32)
            middle = np.random.randint(0, 10, size=(n_sets-int(42*2*n_sets/100), setsize))
            
            samples = np.concatenate((lower, middle, upper), axis=0)
            all_counts = np.array(list(map(lambda t: np.bincount(t, minlength=10), samples)))
            labels = list(np.median(samples.astype(np.float32), axis=1))
            for counts in all_counts:
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
        elif label_type == 'min_over_max':
            label_dict = {k:[] for k in np.unique([min(i,j)/max(i,j) \
                          for i in range(10) for j in range(10) \
                          if i != 0 or j != 0])}
            for i in range(10):
                for j in range(10):
                    if i != 0 or j != 0:
                        label_dict[min(i,j)/max(i,j)] += [(min(i,j),max(i,j))]
            new_dict = {} 
            for i,k in enumerate(sorted(label_dict.keys())):
                new_dict[i] = np.array(list(set(label_dict[k])))
            del label_dict
            counts = np.zeros(10, dtype=np.int32)
            for s in range(n_sets):
                idx = s%len(new_dict)
                min_, max_ = new_dict[idx][np.random.choice(len(new_dict[idx]))]
                labels.append(min_ / max_)
                elements = np.random.randint(2, setsize + 1)
                counts *= 0
                counts[min_] += 1
                counts[max_] += 1
                classes = np.random.choice(range(min_,max_+1), elements-2)
                counts += np.bincount(classes,minlength=10)
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
        elif label_type == 'max_minus_min':
            label_dict = {k:[] for k in np.unique([max(i,j)-min(i,j) \
                          for i in range(10) for j in range(10)])}
            for i in range(10):
                for j in range(10):
                    if i != 0 or j != 0:
                        label_dict[max(i,j)-min(i,j)] += [(min(i,j),max(i,j))]
            new_dict = {} 
            for i,k in enumerate(sorted(label_dict.keys())):
                new_dict[i] = np.array(list(set(label_dict[k])))
            del label_dict
            counts = np.zeros(10, dtype=np.int32)
            for s in range(n_sets):
                idx = s%len(new_dict)
                min_, max_ = new_dict[idx][np.random.choice(len(new_dict[idx]))]
                labels.append(max_ - min_)
                elements = np.random.randint(2, setsize + 1)
                counts *= 0
                counts[min_] += 1
                counts[max_] += 1
                classes = np.random.choice(range(min_,max_+1), elements-2)
                counts += np.bincount(classes,minlength=10)
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
        elif label_type == 'median_sum':
            lower = np.round(np.random.beta(0.05 , 0.25,size=(int(42*n_sets/100), \
                                            setsize))*9).astype(np.int32)
            upper = np.round(np.random.beta(0.25, 0.6,size=(int(42*n_sets/100), \
                                            setsize))*9).astype(np.int32)
            middle = np.random.randint(0, 10, size=(n_sets-int(42*2*n_sets/100), setsize))
            
            samples = np.concatenate((lower, middle, upper), axis=0)
            all_counts = np.array(list(map(lambda t: np.bincount(t, minlength=10), samples)))
            labels = list(np.sum(samples*(samples >= np.median(samples,axis=1, keepdims=True)), axis=1))
            for counts in all_counts:
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
        elif label_type == 'quantile':
            lower = np.round(np.random.beta(0.12 ,0.4,size=(int(42*n_sets/100), \
                                            setsize))*9).astype(np.int32)
            upper = np.round(np.random.beta(0.12, 0.35,size=(int(42*n_sets/100), \
                                            setsize))*9).astype(np.int32)
            middle = np.random.randint(0, 10, size=(n_sets-int(42*2*n_sets/100), setsize))
            
            samples = np.concatenate((lower, middle, upper), axis=0)
            all_counts = np.array(list(map(lambda t: np.bincount(t, minlength=10), samples)))
            labels = list(np.quantile(samples.astype(np.float32), 0.75, interpolation='midpoint', axis=1))
            for counts in all_counts:
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
        elif label_type == 'std':
            join = np.array(np.meshgrid(np.linspace(0, 10, 20), \
                                        np.linspace(0, 4.5, 20))).T.reshape(-1, 2)
            assert len(join) < n_sets
            samples = []
            for i,s in enumerate(range(n_sets)):
                elements = setsize
                #elements = np.random.choice(2, setsize+1)
                join_idx = i%len(join)
                sample = np.random.normal(size=elements,loc=join[join_idx,0], \
                                          scale=join[join_idx,1])
                sample = np.clip(np.round(sample), 0, 9)
                sample = sample.astype(np.int32)
                samples.append(sample)
            all_counts = np.array(list(map(lambda t: np.bincount(t, minlength=10), samples)))
            labels = [np.std(sample) for sample in samples]
            for counts in all_counts:
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
        elif label_type == 'skew':
            lower = np.round(np.random.beta(0.12 ,0.4,size=(int(42*n_sets/100), \
                                            setsize))*9).astype(np.int32)
            upper = np.round(np.random.beta(0.12, 0.35,size=(int(42*n_sets/100), \
                                            setsize))*9).astype(np.int32)
            middle = np.random.randint(0, 10, size=(n_sets-int(42*2*n_sets/100), setsize))
            
            samples = np.concatenate((lower, middle, upper), axis=0)
            all_counts = np.array(list(map(lambda t: np.bincount(t, minlength=10), samples)))
            std  = np.maximum(np.std(samples, axis=1, keepdims=True), 1e-5)
            mean = np.mean(samples, axis=1, keepdims=True)
            skew = np.mean((samples - mean)**3 / std**3, axis=1)
            labels = list(skew)
            for counts in all_counts:
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
        elif label_type == 'kurtosis':
            lower = np.round(np.random.beta(0.12 ,0.4,size=(int(42*n_sets/100), \
                                            setsize))*9).astype(np.int32)
            upper = np.round(np.random.beta(0.12, 0.35,size=(int(42*n_sets/100), \
                                            setsize))*9).astype(np.int32)
            middle = np.random.randint(0, 10, size=(n_sets-int(42*2*n_sets/100), setsize))
            
            samples = np.concatenate((lower, middle, upper), axis=0)
            all_counts = np.array(list(map(lambda t: np.bincount(t, minlength=10), samples)))
            std  = np.maximum(np.std(samples, axis=1, keepdims=True), 1e-5)
            mean = np.mean(samples, axis=1, keepdims=True)
            kurtosis = np.mean((samples - mean)**4 / std**4, axis=1)
            labels = list(kurtosis)
            for counts in all_counts:
                X = []
                for i,c in enumerate(counts): 
                    if c > 0:
                        X += [np.random.choice(class_dict[i], c)]
                sets.append(np.concatenate(X))
    else:
        for s in range(n_sets):
            elements = np.random.randint(1, setsize + 1)
            idxs = np.random.choice(dataset_idx, elements)
            sets.append(idxs)
            if label_type == 'max':
                labels.append( np.max(instance_labels[idxs]))
            elif label_type == 'min':
                labels.append( np.min(instance_labels[idxs]))
            elif label_type == 'count':
                labels.append( len(instance_labels[idxs]))
            elif label_type == 'inv_count':
                labels.append( 1./ len(instance_labels[idxs]))
            elif label_type == 'sum':
                labels.append( np.sum(instance_labels[idxs]))
            elif label_type == 'mean':
                labels.append( np.mean(instance_labels[idxs]))
            elif label_type == 'median':
                labels.append( np.median(instance_labels[idxs]))
            elif label_type == 'min_over_max':
                labels.append( np.min(instance_labels[idxs]) / np.maximum(1.0, np.max(instance_labels[idxs])))
            elif label_type == 'max_minus_min':
                labels.append( np.max(instance_labels[idxs])-np.min(instance_labels[idxs]))
            elif label_type == 'median_sum':
                median = np.median( instance_labels[idxs] )
                labels.append( np.sum(instance_labels[idxs][instance_labels[idxs] >= median]))
            elif label_type == 'quantile':
                labels.append( np.quantile(instance_labels[idxs], 0.75, interpolation='midpoint') )
            elif label_type == 'std':
                labels.append(np.std( instance_labels[idxs] ))
            elif label_type == 'skew':
                std  = np.maximum(np.std(instance_labels[idxs]), 1e-5)
                mean = np.mean(instance_labels[idxs])
                skew = np.mean((instance_labels[idxs]- mean)**3 / std**3)
                labels.append(skew)
            elif label_type == 'kurtosis':
                std  = np.maximum(np.std(instance_labels[idxs]), 1e-5)
                mean = np.mean(instance_labels[idxs])
                kurtosis = np.mean((instance_labels[idxs] - mean)**4 / std**4)
                labels.append(kurtosis)
            ### TODO ALL THE OTHER FUNCTIONS
            else:
                raise Exception('Incorrect labeling string.')
    return np.array(sets), np.array(labels).astype(np.float32)

if __name__ == '__main__':
    fpath = 'data'
    MAX_SET_SIZE = 10 

    parser = argparse.ArgumentParser(description='Generate datasets')
    parser.add_argument('--label', required=True, type=str,
                        help='Label for the sets.')
    parser.add_argument('--seed',default=42, type=int,
                        help='Seed for generating the data.')
    parser.add_argument('--bias', default=False,
                        action='store_true',
                        help='Generate biased dataset.')

    args = parser.parse_args()
    bias = 'biased' if args.bias else 'unbiased'
    assert args.label in get_labels()
    np.random.seed(args.seed)
    os.makedirs(fpath, exist_ok=True)
    data_train = datasets.MNIST(fpath, train=True, download=True)
    data_test  = datasets.MNIST(fpath, train=False, download=True)

    
    X_train_raw, y_train_raw = torch_to_np(data_train.data), torch_to_np(data_train.targets)
    X_test_raw,  y_test_raw  = torch_to_np(data_test.data), torch_to_np(data_test.targets)
   
    X_train_raw = (X_train_raw/255.).astype(np.float32)
    X_test_raw  = (X_test_raw/255.).astype(np.float32)
    X = np.vstack((X_train_raw, X_test_raw))
    y = np.concatenate((y_train_raw, y_test_raw))
    np.save(os.path.join(fpath, 'embeddings.npy'), {'X':X, 'y':y})
    fpath = os.path.join(fpath, args.label, bias, str(args.seed))
    os.makedirs(fpath, exist_ok=True)
    validation_percentage = 0.2
    train_idx = np.arange(len(X_train_raw))
    valid_idx = np.random.choice(train_idx, int(0.2*len(train_idx)), replace=False)
    train_idx = np.array(list(set(train_idx) - set(valid_idx)))
    train_idx = np.sort(train_idx)
    valid_idx = np.sort(valid_idx)
    test_idx  = np.arange(len(X_test_raw)) + len(X_train_raw)
    np.save(os.path.join(fpath, 'train_valid_test_idx.npy'), [train_idx, valid_idx, test_idx])


    print('Generating training set...')
    X_train, y_train = generate_dataset(train_idx, y, int(1e5), MAX_SET_SIZE, args.label, args.bias)
    np.save(os.path.join(fpath, 'X_train_{:02d}.npy'.format(MAX_SET_SIZE)), {'X': X_train, 'y': y_train})
    print('Generating validation set...')
    X_valid, y_valid = generate_dataset(valid_idx, y, int(2e4), MAX_SET_SIZE, args.label, args.bias)
    np.save(os.path.join(fpath, 'X_valid_{:02d}.npy'.format(MAX_SET_SIZE)), {'X': X_valid, 'y': y_valid})
    pbar = tqdm(range(5,51,5))
    #pbar = tqdm([5,10])
    for set_size in pbar:
        pbar.set_description('Generating test ({:02d})'.format(set_size))
        X_test, y_test = generate_dataset(test_idx, y, int(1e5), set_size, args.label, args.bias)
        np.save(os.path.join(fpath, 'X_test_{:02d}.npy'.format(set_size)), {'X':X_test, 'y':y_test})
