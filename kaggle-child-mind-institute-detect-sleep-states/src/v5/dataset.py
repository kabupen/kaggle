import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import DataLoader, Dataset
import gc
from tqdm import tqdm
from torchvision.transforms.functional import resize

from utils import random_crop 

def pad_if_needed(x: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
    if len(x) == max_len:
        return x
    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0, 0) for _ in range(n_dim - 1)]
    return np.pad(x, pad_width=pad_widths, mode="constant", constant_values=pad_value)


def load_features(data_path, series_id_list, feature_names):
    """
    - 時系列データを dict で保持する
    - Dataset 内で crop して固定長にするので、この段階では可変長でok
    """
    features = {}

    for series_id in series_id_list:
        this_feat = []
        for feat_name in feature_names:
            this_feat.append(np.load(f"{data_path}/{series_id}/{feat_name}.npy"))
        features[series_id] = np.stack(this_feat, axis=1)
    
    return features

def load_chunk_features(data_path, n_frames, series_id_list, feature_names):
    """
    - 時系列データを固定長に変換して、dict で保持する
    """
    features = {}
    for series_id in series_id_list:
        this_feat = []
        for feat_name in feature_names:
            this_feat.append(np.load(f"{data_path}/{series_id}/{feat_name}.npy"))
        this_feat = np.stack(this_feat, axis=1)

        num_chunks = int(len(this_feat)//n_frames) + 1
        for idx in range(num_chunks):
            chunk_feat = this_feat[idx*n_frames:(idx+1)*n_frames]
            chunk_feat = pad_if_needed(chunk_feat, n_frames, pad_value=0)
            features[f"{series_id}_{idx:07}"] = chunk_feat
    
    return features

def get_label(df, n_frames, start_idx, end_idx):
    """
    Args:
        - df : ピボットテーブル [series_id, wakeup, onset]
    """

    # 説明変数と同じ時間領域の情報を取得
    df = df.query("@start_idx <= wakeup & onset <= @end_idx")

    # dim=0(sleep), 1(onset), 2(wakeup)
    label = np.zeros((n_frames, 3))
    for onset, wakeup in df[["onset", "wakeup"]].values:

        onset =  int(onset - start_idx)
        wakeup =  int(wakeup - start_idx)

        if onset >= 0 and onset < n_frames:
            label[onset, 1] = 1
        if wakeup >=0 and wakeup < n_frames:
            label[wakeup, 2] = 1

        onset = max(0, onset)
        wakeup = min(n_frames, wakeup)
        label[onset:wakeup, 0] = 1 # sleep

    return label

def gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))  # type: ignore
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], gaussian_kernel(offset, sigma), mode="same")

    return label

def negative_sampling(this_event_df: pd.DataFrame, num_steps: int) -> int:
    """negative sampling

    Args:
        this_event_df (pd.DataFrame): event df
        num_steps (int): number of steps in this series

    Returns:
        int: negative sample position
    """
    # onsetとwakupを除いた範囲からランダムにサンプリング
    positive_positions = set(this_event_df[["onset", "wakeup"]].to_numpy().flatten().tolist())
    negative_positions = list(set(range(num_steps)) - positive_positions)
    return random.sample(negative_positions, 1)[0]

class SleepTrainDataset(Dataset):

    def __init__(self, features, df_events, n_frames, cfg):

        self.features = features
        self.n_frames = n_frames # 使用する時系列データの個数 (/5sec)
        self.cfg = cfg

        # events.csv pivot table
        self.df_pivot = df_events.pivot(index=["series_id", "night"], columns="event", values="step").dropna().reset_index()
    
    def __len__(self):
        return len(self.df_pivot)
    
    def __getitem__(self, idx):

        event_str = np.random.choice(["onset", "wakeup"], p=[0.5, 0.5])
        pos = self.df_pivot.loc[idx, event_str] # 該当の event("onset"/"wakeup") の step 数

        this_series_id = self.df_pivot.loc[idx, "series_id"]
        this_event = self.df_pivot[self.df_pivot["series_id"]==this_series_id].reset_index(drop=True)
        this_feat = self.features[this_series_id] # [#時系列長, #特徴量]

        n_steps = this_feat.shape[0]

        # negative sampling
        if random.random() < 0.5:
            pos = negative_sampling(this_event, n_steps)

        # crop
        # - 目的変数の位置 (pos) を含むように特徴量を cropする
        # - 該当のデータが想定している時系列長を持っているか確認、crop or padding 処理を実行する
        if n_steps > self.n_frames:
            start_idx, end_idx = random_crop(pos, self.n_frames, n_steps)
            this_feat = this_feat[start_idx:end_idx]
        else:
            start_idx, end_idx = 0, self.n_frames
            this_feat = pad_if_needed(this_feat, self.n_frames, pad_value=0)
        
        this_feat = torch.FloatTensor(this_feat.T) # [#特徴量, self.n_frames]

        # upsample
        # this_feat = this_feat.unsqueeze(0) # [1, #特徴量, self.n_frames]
        # this_feat = resize(
        #     this_feat,
        #     size=[self.num_features, self.upsampled_num_frames],
        #     antialias=False,
        # ).squeeze(0)

        # gaussian label
        #   - onset/wakeup 正解ラベルに幅をもたせている
        label = get_label(this_event, self.n_frames, start_idx, end_idx)
        label[:, [1, 2]] = gaussian_label(
            label[:, [1, 2]], offset=self.cfg.dataset.label_offset, sigma=self.cfg.dataset.label_sigma
        )

        return {
            "series_id": this_series_id,
            "X": this_feat,
            "y_true": torch.tensor(label)
        }


class SleepValidDataset(Dataset):

    def __init__(self, chunk_features, df_events, n_frames):
        """
        Args:
            chunk_features
            df_events
            n_frames
        """
        self.chunk_features = chunk_features
        self.keys = list(self.chunk_features.keys()) # {series_id}_{index}
        self.n_frames = n_frames # 使用する時系列データの個数 (/5sec)
        
        # events.csv pivot table
        self.df_pivot = df_events.pivot(index=["series_id", "night"], columns="event", values="step").dropna().reset_index()
    
    def __len__(self):
        return len(self.df_pivot)
    
    def __getitem__(self, idx):

        key = self.keys[idx]
        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        
        this_feat = self.chunk_features[key]
        this_feat = torch.FloatTensor(this_feat.T) # [1, #特徴量, #時系列]

        # # upsample
        # this_feat = this_feat.unsqueeze(1)
        # feature = resize(
        #     feature,
        #     size=[],
        #     antialias=False
        # ).squeeze(0)

        this_event = self.df_pivot[self.df_pivot["series_id"]==series_id].reset_index(drop=True)

        start_idx = chunk_id * self.n_frames
        end_idx = start_idx + self.n_frames

        # label
        label = get_label(this_event, self.n_frames, start_idx, end_idx)

        return {
            "key": key,
            "X" : this_feat, # [#特徴量, #時系列]
            "y_true" : torch.tensor(label) # [#時系列, 3]
        }


class SleepTestDataset(Dataset):

    def __init__(self, chunk_features, n_frames):

        self.chunk_features = chunk_features
        self.keys = list(self.chunk_features.keys()) # {series_id}_{index}
        self.n_frames = n_frames # 使用する時系列データの個数 (/5sec)
    
    def __len__(self):
        return len(self.chunk_features)
    
    def __getitem__(self, idx):

        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T)
        # feature.unsqueeze(0) # [1, #特徴量, #時系列]
        # feature = resize(
        #     feature,
        #     size=[],
        #     antialias=False
        # ).squeeze(0)

        return {
            "key": key,
            "X": feature
        }