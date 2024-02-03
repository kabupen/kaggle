import pandas as pd
import pickle
from sklearn.model_selection import KFold

if __name__ == "__main__" :

    df_events = pd.read_csv("/ebs03/child-mind-institute-detect-sleep-states/data/train_events.csv")
    series_id_list = df_events["series_id"].unique()
    kf = KFold(n_splits=5)

    fold_id_dict = {}
    for idx_fold, (train_id, valid_id) in enumerate(kf.split(series_id_list)):
        train_series_id = series_id_list[train_id]
        valid_series_id = series_id_list[valid_id]

        fold_id_dict[f"f{idx_fold}"] = {
            "train" : train_series_id,
            "valid" : valid_series_id
        }
    
    with open("/ebs03/child-mind-institute-detect-sleep-states/output/v5/kfold/kfold_series_id.pkl", "wb") as f:
        pickle.dump(fold_id_dict, f)