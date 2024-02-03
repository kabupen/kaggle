import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from copy import deepcopy
import hydra

from metric import score
from feature import get_events
from mlflow_helper import MlflowHelper

@hydra.main(config_path="../../config", config_name="")
def main(cfg):    

    # mlflow
    mlflow_helper = MlflowHelper(cfg.mlflow_output_path, cfg.experiment_name)
    mlflow_helper.create_run()

    # log parameters
    mlflow_helper.log_params(cfg.lgbm_params)

    score_list = []
    for idx_fold in range(cfg.num_folds):
   
        print(">>> data load")
        train_data = pd.read_pickle("/ebs03/child-mind-institute-detect-sleep-states/output/features/train_series_v1.pkl")
        X_train = train_data.drop(columns={"series_id", "timestamp", "step", "awake"})
        y_train = train_data["awake"]

        valid_data = pd.read_pickle("/ebs03/child-mind-institute-detect-sleep-states/output/features/valid_series_v1.pkl")
        X_valid = valid_data.drop(columns={"series_id", "timestamp", "step", "awake"})
        y_valid = valid_data["awake"]

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

        # scoring
        # (1) solution (from train_events.csv)
        solution = pd.read_csv('/ebs03/child-mind-institute-detect-sleep-states/data/train_events.csv')
        solution = solution[solution["series_id"].isin(valid_data["series_id"])]

        # (2) submit format 
        X_valid = valid_data.drop(columns={"timestamp", "awake"})
        X_valid.loc[:, "pred"] = model.predict(X_valid)
        X_valid.loc[:, "prob"] = model.predict_proba(X_valid)[:,1]
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