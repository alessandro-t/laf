import argparse
import numpy as np
import torch
import torch_scatter
import os
import sys
from glob import glob
from tqdm.auto import tqdm

from dataset import Dataset
from utils import torch_to_np, np_to_torch, get_labels

sys.path.append('../laf') 
from model import LAFLayer

class Model(torch.nn.Module):
    def __init__(self, embeddings, units, mode):
        super(Model, self).__init__()
        assert mode in ['laf']
        self.mode  = mode
        self.units = max(3*(units//3), 3)
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings , freeze=True)
        self.linear_1 = torch.nn.Linear(784, 300) 
        self.linear_2 = torch.nn.Linear(300, 100)
        self.linear_3 = torch.nn.Linear(100, 30)
        if self.mode == 'laf':
            w1 = np.random.uniform(size=(8, self.units)).astype(np.float32)
            w2 = np.random.normal(size=(4, self.units), loc=0.0, scale=0.01).astype(np.float32)
            w = np.vstack((w1,w2))
            w = torch.tensor(w)
            self.laf = LAFLayer(weights=w)
            self.linear_4 = torch.nn.Linear(30*self.units,1000)
            self.linear_5 = torch.nn.Linear(1000,100)
            self.linear_6 = torch.nn.Linear(100,1)
        print(self)

    def forward(self, x, idx):
        x = self.embeddings(x)
        x = self.linear_1(x)
        x = torch.tanh(x)
        x = self.linear_2(x)
        x = torch.tanh(x)
        x = self.linear_3(x)  
        x = torch.sigmoid(x)
        x = self.laf(x, idx)
        x = x.view((-1, 30*self.units))
        x = self.linear_4(x)
        x = torch.tanh(x)
        x = self.linear_5(x)
        x = torch.tanh(x)
        x = self.linear_6(x)
        return x

if __name__ == '__main__':
    DEVICE = 'cuda:0'
    EPOCHS = 100
    BATCHSIZE = 64
    fpath = 'data'
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--model', required=True, type=str,
                        help='Model type.')
    parser.add_argument('--units', default=60, type=int,
                        help='Model units.')
    parser.add_argument('--label', required=True, type=str,
                        help='Label for the sets.')
    parser.add_argument('--seed',default=42, type=int,
                        help='Seed used for generating the data.')
    parser.add_argument('--setsize',default=10, type=int,
                        help='Set size used for generating the data.')
    parser.add_argument('--bias', default=False,
                        action='store_true',
                        help='Biased dataset.')
    parser.add_argument('--run',default=0, type=int,
                        help='Run.')

    args = parser.parse_args()
    bias = 'biased' if args.bias else 'unbiased'
    assert args.label in get_labels()

    data_path = os.path.join(fpath, args.label, bias, str(args.seed)) 
    os.makedirs(os.path.join(data_path, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'predictions', args.model, 'biased'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'predictions', args.model, 'unbiased'), exist_ok=True)

    train = Dataset(os.path.join( data_path, 'X_train_{:d}.npy'.format(args.setsize)))
    valid = Dataset(os.path.join( data_path, 'X_valid_{:d}.npy'.format(args.setsize)))

    embs = np.load('data/embeddings.npy', allow_pickle=True)[()]
    embs = embs['X'].reshape((len(embs['X']),-1))
    embs = torch.tensor(embs)
    model = Model(embs, units=args.units, mode=args.model).to(DEVICE)


    loss_fn = torch.nn.MSELoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, \
                                                           factor=0.5, min_lr=0.000001) 
    pbar = tqdm(range(EPOCHS))
    valid_loss = np.infty
    best_valid_loss = valid_loss
    model_path = os.path.join(data_path, 'checkpoints', \
                              '{}_{:04d}_{:02d}.mdl'.format(args.model, args.units, args.run))
    for epoch in pbar:
        train_loss = []
        model.train()
        print(model.laf.weights[:,0])
        for xb,sb,yb in train.get_minibatches(BATCHSIZE, use_torch=True, device=DEVICE):
            optimizer.zero_grad() 
            pred = model(xb, sb)
            loss = loss_fn(yb.view(-1, 1), pred.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss.append(torch_to_np(loss))
            pbar.set_description('TL: {:.3f} - VL: {:.3f}'.format(np.mean(train_loss[-50:]), valid_loss))
        model.eval()
        valid_loss = []
        counter = 0.0
        for xb,sb,yb in valid.get_minibatches(BATCHSIZE, use_torch=True, device=DEVICE):
            pred = model(xb, sb)
            loss = loss_fn(yb.view(-1, 1), pred.view(-1, 1))
            valid_loss.append(len(yb)*torch_to_np(loss))
            counter += len(yb)
        valid_loss = sum(valid_loss) / counter
        scheduler.step(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)
    print('Reloading best valid model...')
    model.load_state_dict(torch.load(model_path)) 
    model.eval()
    print('Saving all predictions...')
    for type_ in ['biased', 'unbiased']:
        all_paths = os.path.join(fpath, args.label, type_, str(args.seed), 'X_*.npy')
        save_path = os.path.join(data_path, 'predictions', args.model, type_)
        for path in tqdm(glob(all_paths)):
            test = Dataset(path)
            _, fname = os.path.split(path)
            fname = '.'.join(fname.split('.')[:-1]) + '_{:04d}_{:02d}.npy'.format(args.units, args.run)
            fname = os.path.join(save_path, fname)
            preds = []
            for xb,sb,yb in test.get_minibatches(BATCHSIZE, use_torch=True, device=DEVICE, drop_reminder=False, shuffle=False):
                pred = model(xb, sb)
                preds += [torch_to_np(pred).ravel()]
            preds = np.concatenate(preds)
            np.save(fname, preds)

