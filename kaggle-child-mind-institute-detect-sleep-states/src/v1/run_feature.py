import pandas as pd
from feature import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

if __name__ == "__main__" : 

    # dataset = Dataset("/ebs03/child-mind-institute-detect-sleep-states/data/", "train")
    dataset = Dataset("/ebs03/child-mind-institute-detect-sleep-states/datasets/detect-sleep-states-memory-decrease", "train") # メモリ削減データセット

    series_id_list = dataset.df_series["series_id"].unique()

    # テスト用に適当に：
    # train_series_id = ['038441c925bb', '03d92c9f6f8a', '0402a003dae9', '04f547b8017d',]

    # prepare
    df_full_series = pd.DataFrame()
    for series_id in tqdm(series_id_list):
        df_series = dataset.make_series_features(series_id)
        df_full_series = pd.concat([df_full_series, df_series], axis=0)
    
    df_full_series.to_pickle(f"/ebs03/child-mind-institute-detect-sleep-states/output/features/train_series_v1.pkl")