import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import copy
import logging

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    return df['pred'].corr(df['label']), df['pred'].corr(df['label'], method='spearman')

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    p = int(0.025*N)
    filtered = indices[p:-p]
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[filtered] = True
    return mask, x[mask]

def drop_na(x):
    mask = ~x.isnan()
    return mask, x[mask]

def zscore(x):
    return (x - x.mean()).div(x.std())

class DailyBatchSampler:
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.daily_count = pd.Series(index=data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        indices = np.arange(len(self.daily_count))
        if self.shuffle: np.random.shuffle(indices)
        for i in indices:
            yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])

    def __len__(self):
        return len(self.data_source)

class Trainer:
    def __init__(self, config):
        self.config = config
        t = config['training']
        self.device = torch.device(f"cuda:{t['gpu']}" if torch.cuda.is_available() else "cpu")
        self.seed = t.get('seed')
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        
    def _dataloader(self, data, shuffle=True):
        return DataLoader(data, sampler=DailyBatchSampler(data, shuffle), drop_last=shuffle)
     

    def train_epoch(self, model, optimizer, loader):
        model.train()
        losses = []
        for i, data in enumerate(loader):
            data = torch.squeeze(data, dim=0) # stks, seq_len, input_size
            feature = data[:, :, :-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            
            mask, label = drop_extreme(label)
            feature, label = feature[mask], zscore(label)
            
            pred = model(feature.float())
            loss = ((pred - label)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(loss.item())
           
        return float(np.mean(losses))
     
    def test(self, model, loader):
        model.eval()
        labels, preds, ic_list, ric_list = [], [], [], []
        with torch.no_grad():
            for data in loader:
                data = torch.squeeze(data, dim=0)
                feature = data[:, :, :-1].to(self.device)
                label = data[:, -1, -1]
                
                # mask, label = drop_na(label)
                # label = zscore(label)
                pred = model(feature.float()).cpu().numpy()
                preds.append(pred.ravel())
                labels.append(label.cpu().numpy().ravel())
                
                ic, ric = calc_ic(pred, label.numpy())
                ic_list.append(ic)
                ric_list.append(ric)
        
        score = pd.DataFrame(data = {'score':np.concatenate(preds),'label':np.concatenate(labels)}, index=loader.dataset.get_index())
        metrics = {
            'IC': np.mean(ic_list), 'ICIR': np.mean(ic_list)/np.std(ic_list),
            'RIC': np.mean(ric_list), 'RICIR': np.mean(ric_list)/np.std(ric_list)
        }
        save_dir = Path(f'scores/')
        save_dir.mkdir(exist_ok=True, parents=True)
        score.to_csv(save_dir / f"{self.config['data']['universe']}_{self.seed}.csv")
        return score, metrics
    
    def fit(self, model, dl_train, dl_valid=None):
        optimizer = optim.Adam(model.parameters(), self.config['training']['lr'])
        train_loader = self._dataloader(dl_train, shuffle=True)
        
        for epoch in range(self.config['training']['n_epoch']):
            train_loss = self.train_epoch(model, optimizer, train_loader)
            
            if dl_valid:
                _, metrics = self.test(model, self._dataloader(dl_valid, shuffle=False))
                logging.info(f"Epoch {epoch}, loss: {train_loss:.4f}, IC: {metrics['IC']:.4f}, ICIR: {metrics['ICIR']:.3f}")
            else:
                logging.info(f"Epoch {epoch}, loss: {train_loss:.4f}")

            if train_loss <= self.config['training']['train_stop_loss_thred']:
                Path(f"{self.config['training']['save_path']}").mkdir(exist_ok=True, parents=True)
                torch.save(model.state_dict(), f"{self.config['training']['save_path']}/{self.config['data']['universe']}_{self.seed}.pkl")
                break


