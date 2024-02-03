import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

# local
from feature import Dataset

if __name__ == "__main__" : 

    random.seed(42)

    dataset = Dataset("/ebs03/child-mind-institute-detect-sleep-states/datasets/detect-sleep-states-memory-decrease/", "train")

    series_id_list = dataset.df_full_series["id_map"].unique()

    # テスト用に適当に：
    # train_series_id = ['038441c925bb', '03d92c9f6f8a', '0402a003dae9', '04f547b8017d',]

    # でかすぎるのでここで分割（= Group k-fold）
    random.shuffle(series_id_list)
    num_fold = 5
    fold_table = {}
    for idx_fold in range(num_fold):
        fold_table[idx_fold] = series_id_list[idx_fold * len(series_id_list)//num_fold : (idx_fold+1) * len(series_id_list)//num_fold ]

    # check 
    total_chunk = 0
    for idx, l in fold_table.items():
        total_chunk += len(l)
        print(f"fold = {idx} {len(l)}")
    assert len(series_id_list) == total_chunk, "The chunk size isn't consistent..."

    # prepare
    df_full_series = pd.DataFrame()
    for idx_fold, fold_series_id_list in fold_table.items():
        for series_id in tqdm(fold_series_id_list):
            df_series = dataset.make_series_features(series_id)
            df_series["fold"] = idx_fold
            df_full_series = pd.concat([df_full_series, df_series], axis=0)
    
    df_full_series.to_pickle(f"/ebs03/child-mind-institute-detect-sleep-states/output/features/v3/train_series_multi.pkl")