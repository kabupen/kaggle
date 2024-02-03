import pandas as pd
from sklearn.model_selection import KFold

if __name__ == "__main__":

    ROOT_PATH = f"/ebs03/child-mind-institute-detect-sleep-states/output/features/"
    for mode_str in ["train", "valid"] :
        df_data = pd.read_pickle(f"{ROOT_PATH}/{mode_str}_series_v1.pkl") 
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for idx_fold, (_, valid_idx) in enumerate(kf.split(df_data)):
            df_data.loc[valid_idx, 'fold'] = idx_fold

        df_data.to_pickle(f"{ROOT_PATH}/{mode_str}_series_v1_fold.pkl")