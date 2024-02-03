import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from tqdm import tqdm

if __name__ == "__main__":

    ROOT_PATH = f"/ebs03/child-mind-institute-detect-sleep-states/output/features/v2/"

    df_data = pd.read_pickle(f"{ROOT_PATH}/train_series.pkl") 

    # (1) K-Fold 
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # for idx_fold, (_, valid_idx) in tqdm(enumerate(kf.split(df_data))):
    #     print(valid_idx, len(valid_idx))
    #     df_data.loc[valid_idx, 'fold'] = idx_fold

    # df_data.to_pickle(f"{ROOT_PATH}/train_series_fold.pkl")
    
    # (2) Group K-Fold 
    print(df_data.columns)
    kf = GroupKFold(n_splits=5)
    X = df_data.drop(columns={"id_map", "timestamp", "step", "awake"})
    y = df_data["awake"]
    for idx_fold, (_, valid_idx) in enumerate(kf.split(X, y, df_data["id_map"])):
        # print(valid_idx, len(valid_idx))
        length = len(valid_idx)
        df_data.loc[valid_idx[0:length//10], 'fold'] = idx_fold

    df_data.to_pickle(f"{ROOT_PATH}/train_series_fold.pkl")