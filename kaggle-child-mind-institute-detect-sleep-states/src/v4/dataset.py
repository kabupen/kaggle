import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import gc
from tqdm import tqdm

class SleepDataset(Dataset):
    def __init__(self, df_series, df_events = None, mode="train", id_key="id_map"):
        
        self.mode = mode
        
        self.series_id_list = df_series[id_key].unique()
        self.sample_freq = 12 # 5*12 = 60 sec 単位のデータにする
        
        self.data, self.step = [], []
        for idx, series_id  in tqdm(enumerate(self.series_id_list), total=len(self.series_id_list)):
            tmp_data = df_series[df_series[id_key]==series_id].copy().reset_index(drop=True)[["anglez", "enmo"]].values
            tmp_step = df_series[df_series[id_key]==series_id].copy().reset_index(drop=True)[["step"]]
            self.data.append(tmp_data)
            self.step.append(tmp_step)
            # if idx > 20: break
        del df_series, tmp_data, tmp_step

        if self.mode == "train":
            self.targets = []
            for idx, series_id  in tqdm(enumerate(self.series_id_list), total=len(self.series_id_list)):
                tmp = df_events[df_events[id_key]==series_id].copy().reset_index(drop=True)[["event_1", "event_2"]].values
                self.targets.append(tmp)
                # if idx > 20: break
            del df_events


    def generate_features(self, feat):
        """
        Args:
            - [#時系列/5sec,]
        Returns:
            - [#時系列/60sec, #特徴量]
        """
        
        # sample_freq 単位に reshape できない場合に備えて、zero-padding する
        if len(feat) % self.sample_freq != 0:
            feat = np.concatenate([
                feat,
                np.zeros(self.sample_freq - len(feat) % self.sample_freq)
            ], axis=0)
        
        feat = np.reshape(feat, (-1, self.sample_freq)) # [x, 12]
        feat_mean = np.mean(feat, axis=1)               # [x,]
        feat_std = np.std(feat, axis=1)
        feat_median = np.median(feat, axis=1)
        feat_max = np.max(feat, axis=1)
        feat_min = np.min(feat, axis=1)

        # [1, x, 5] -> [x, 5]
        return np.dstack([
            feat_mean,
            feat_std,
            feat_median,
            feat_max,
            feat_min])[0]

    def arrange_targets(self, target):
        
        # sample_freq 単位に reshape できない場合に備えて、zero-padding する
        if len(target) % self.sample_freq != 0:
            target = np.concatenate([
                target,
                np.zeros(self.sample_freq - len(target) % self.sample_freq)
            ], axis=0)
            
        target = np.reshape(target, (-1, self.sample_freq))
        target_mean = np.mean(target, axis=1) # [時系列長,]
        return np.expand_dims(target_mean, axis=1)
    
    # def gauss(self,n=SIGMA,sigma=SIGMA*0.15):
    #     # guassian distribution function
    #     r = range(-int(n/2),int(n/2)+1)
    #     return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        index : series_id
        """

        X = self.data[index]    # [時系列長, 2]
        gc.collect()
        
        X = np.concatenate([
            self.generate_features(X[:, idx_feat]) for idx_feat in range(X.shape[1])
            ], axis=-1)
        
        X = torch.from_numpy(X)
        gc.collect()

        if self.mode == "test" : 
            return X

        y = self.targets[index] # [時系列長, 2]
        y = np.concatenate([
            self.arrange_targets(y[:, idx_target]) for idx_target in range(y.shape[1])
            ], axis=-1)
        gc.collect()
        
        # y = normalize(torch.from_numpy(y))
        y = torch.from_numpy(y)
        
        return X, y

def infer_chunk(X, model, max_chunk_size):
    """
    Args:
        X : [#時系列, #特徴量] の次元を持ったセンサーデータ
    
    Returns:
        y : [#時系列, 2] の onset/awake の予測データ
    """
    model.eval()

    # 推論
    seq_len = X.shape[0]
    hidden = None
    pred = torch.zeros((len(X),2)).half()
    for j in range(0, seq_len, max_chunk_size):
        with torch.no_grad():
            y_pred, hidden = model(X[j: j + max_chunk_size].float(), hidden)
            hidden = [h.detach() for h in hidden]
            pred[j : j + max_chunk_size] = y_pred.detach()
        del y_pred
        gc.collect()    
    del hidden, X
    gc.collect()
    
    pred = pred.numpy()
    return pred


def get_submission_events(dataset, model, series_id_list, device, mode="valid", criterion=None):

    max_chunk_size = 10
    min_interval = 30

    model.eval()
    submission = pd.DataFrame()

    loss_list = []
    # dataset で回しているので、先頭にバッチサイズの次元はつかない
    for idx_ds in range(len(dataset)):
        
        if mode == "valid":
            X, y = dataset[idx_ds]
        elif mode == "test":
            X = dataset[idx_ds]
        
        series_id = series_id_list[idx_ds]

        # inference
        seq_len = X.shape[0]
        hidden = None
        y_pred = torch.zeros((len(X), 2)).half()
        for idx_seq in range(0, seq_len, max_chunk_size):
            with torch.no_grad():
                X_chunk = X[idx_seq:idx_seq+max_chunk_size].float().to(device, non_blocking=True)
                pred, hidden = model(X_chunk, hidden)
                hidden = [h.detach() for h in hidden]
                y_pred[idx_seq:idx_seq+max_chunk_size] = pred.detach()
            del pred
        del hidden, X
        gc.collect()
        
        if mode == "valid":
            loss = criterion(y_pred.float().cpu(), y.float().cpu())
            loss_list.append(loss.item())
    
        y_pred = y_pred.numpy()
    
        # arrangements
        days = len(y_pred)/(17280/12)
        scores0 = np.zeros(len(y_pred), dtype=np.float16)
        scores1 = np.zeros(len(y_pred), dtype=np.float16)
        for index in range(len(y_pred)):
            if y_pred[index,0] == max(y_pred[max(0,index-min_interval):index+min_interval,0]):
                scores0[index] = max(y_pred[max(0,index-min_interval):index+min_interval,0])

            if y_pred[index,1] == max(y_pred[max(0,index-min_interval):index+min_interval,1]):
                scores1[index] = max(y_pred[max(0,index-min_interval):index+min_interval,1])

        candidates_onset = np.argsort(scores0)[-max(1,round(days)):]
        candidates_wakeup = np.argsort(scores1)[-max(1,round(days)):]
    
        onset = dataset.step[idx_ds][['step']].iloc[np.clip(candidates_onset*12,0,len(dataset.data[idx_ds])-1)].astype(np.int32)
        onset['event'] = 'onset'
        onset['series_id'] = series_id
        onset['score']= scores0[candidates_onset]
        wakeup = dataset.step[idx_ds][['step']].iloc[np.clip(candidates_wakeup*12,0,len(dataset.data[idx_ds])-1)].astype(np.int32)
        wakeup['event'] = 'wakeup'
        wakeup['series_id'] = series_id
        wakeup['score']= scores1[candidates_wakeup]
    
        submission = pd.concat([submission,onset,wakeup],axis=0)
    
        del onset, wakeup, candidates_onset, candidates_wakeup, scores0, scores1, y_pred, series_id,
        gc.collect()
    
    # finalize
    submission = submission.sort_values(['series_id','step']).reset_index(drop=True)
    submission['row_id'] = submission.index.astype(int)
    submission['score'] = submission['score'].fillna(submission['score'].mean())
    submission = submission[['row_id','series_id','step','event','score']]

    if mode == "valid" : 
        return submission, loss_list
    elif mode == "test" :
        return submission