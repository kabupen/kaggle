import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from copy import deepcopy
import hydra

import sys
sys.path.append("/workspace/child-mind-institute-detect-sleep-states/src/")
from common.metric import score
from common.mlflow_helper import MlflowHelper
from feature import get_events

@hydra.main(config_path="../../config", config_name="")
def main(cfg):    

    # mlflow
    mlflow_helper = MlflowHelper(cfg.mlflow_output_path, cfg.experiment_name)
    mlflow_helper.create_run()

    # log parameters
    mlflow_helper.log_params(cfg.lgbm_params)

    score_list = []
    print(">>> data load")
    df_data = pd.read_pickle("/ebs03/child-mind-institute-detect-sleep-states/output/features/v2/train_series.pkl")

    column_names = {
        'series_id_column_name': 'series_id',
        'time_column_name': 'step',
        'event_column_name': 'event',
        'score_column_name': 'score',
    }
    tolerances = {
        'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360], 
        'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
    }
    
    for idx_fold in range(cfg.num_folds):

        X_train = df_data[df_data["fold"]!=idx_fold].drop(columns={"id_map", "timestamp", "step", "fold", "awake"})
        y_train = df_data[df_data["fold"]!=idx_fold]["awake"]

        X_valid = df_data[df_data["fold"]==idx_fold].drop(columns=["id_map", "timestamp", "step", "fold", "awake"])
        y_valid = df_data[df_data["fold"]==idx_fold]["awake"]

        print(">>> train")
        model = lgb.LGBMClassifier(
            **cfg.lgbm_params
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, verbose=True),
                lgb.log_evaluation(0)]
            )

        del X_train, y_train, X_valid, y_valid


        # scoring
        # (1) solution (from train_events.csv)
        X_valid = df_data[df_data["fold"]==idx_fold]
        
        solution = pd.read_csv('/ebs03/child-mind-institute-detect-sleep-states/data/train_events.csv')
        id_map = pd.read_parquet('/ebs03/child-mind-institute-detect-sleep-states/datasets/detect-sleep-states-memory-decrease/train_id_map.parquet')
        valid_id_map = id_map[id_map["id_map"].isin(X_valid["id_map"].values)]
        solution = solution[solution["series_id"].isin(valid_id_map["series_id"])] # valid の該当イベントだけ抽出

        # (2) submit format 
        pred = model.predict(X_valid.drop(columns={"id_map", "step", "timestamp", "fold", "awake"}))
        prob = model.predict_proba(X_valid.drop(columns={"id_map", "step", "timestamp", "fold", "awake"}))[:,1]
        X_valid.loc[:, "pred"] = pred
        X_valid.loc[:, "prob"] = prob
        X_valid = pd.merge(X_valid, valid_id_map, on="id_map", how="inner") # id_map と series_id の対応付け
        X_valid = X_valid.drop(columns={"id_map"})
        submission = get_events(X_valid)

        # (3) calc score
        pred_score = score(
            solution,
            submission,
            tolerances,
            **column_names
        )
        print("score = ", pred_score)
        score_list.append(pred_score)

        weight_path = f"/tmp/lightgbm_{idx_fold}.pkl"
        with open(weight_path, "wb") as f:
            pickle.dump(model, f)

        mlflow_helper.log_metric(f"f{idx_fold}_score", pred_score)
        mlflow_helper.log_artifact(weight_path)
    
    print(">>>>>>>>")

    mlflow_helper.log_metric("score", np.mean(score_list))
    mlflow_helper.set_terminated()

if __name__ == "__main__" :
    main()