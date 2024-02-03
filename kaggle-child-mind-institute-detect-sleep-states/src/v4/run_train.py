import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from copy import deepcopy
import hydra
import gc

import torch
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler

import sys
sys.path.append("/workspace/child-mind-institute-detect-sleep-states/src/")
from common.metric import score
from common.mlflow_helper import MlflowHelper
from feature import get_events
from model import MultiResidualBiGRU
from dataset import SleepDataset, get_submission_events

@hydra.main(config_path="../../config", config_name="")
def main(cfg):    
    
    max_chunk_size = 15000
    DEVICE = "cuda"

    # evaluation params 
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

    # mlflow
    mlflow_helper = MlflowHelper(cfg.mlflow_output_path, cfg.experiment_name)
    mlflow_helper.create_run()

    # log parameters
    mlflow_helper.log_params(cfg.lgbm_params)

    model = MultiResidualBiGRU(input_size=10, hidden_size=64, out_size=2, n_layers=5).to(DEVICE)

    df_series = pd.read_pickle("/ebs03/child-mind-institute-detect-sleep-states/output/features/v4/prep_train_series.pkl") 
    df_events = pd.read_pickle("/ebs03/child-mind-institute-detect-sleep-states/output/features/v4/prep_train_events.pkl") 

    pred_score_all = []
    for idx_fold in range(1):

        train_dataset = SleepDataset(df_series[df_series["fold"]!=idx_fold], df_events[df_events["fold"]!=idx_fold])
        valid_dataset = SleepDataset(df_series[df_series["fold"]==idx_fold], df_events[df_events["fold"]==idx_fold])
        train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, pin_memory=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0)
        steps = int(len(train_dataset)*0.9)*5
        scheduler = CosineLRScheduler(optimizer, t_initial=steps, warmup_t=int(steps*0.2), warmup_lr_init=1e-6, lr_min=2e-8,)
        criterion = torch.nn.MSELoss()

        for idx_epoch in range(100):

            ### train
            model.train()
            for idx_step, data in enumerate(train_loader):

                optimizer.zero_grad()
                scheduler.step(idx_step)

                X_batch, y_batch = data
                y_pred = torch.zeros(y_batch.shape).to(DEVICE)
                
                bs, seq_len, num_feat = X_batch.shape
                hidden = None
                for idx in range(0, seq_len, max_chunk_size):
                    X_chunk = X_batch[:, idx:idx+max_chunk_size].float().to(DEVICE, non_blocking=True)

                    # foraward
                    pred, hidden = model(X_chunk, hidden)
                    hidden = [ h.detach() for h in hidden]
                    y_pred[:, idx:idx+max_chunk_size] = pred

                    del X_chunk, pred
                    gc.collect()

                loss = criterion(y_pred.float().cpu(), y_batch.float().cpu())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)
                optimizer.step()

                mlflow_helper.running_update("train_loss", loss.item())

                del y_pred, loss, y_batch, X_batch, hidden
                gc.collect()

            ### valid
            
            # solution
            # - オリジナルファイルの train_events.csv を読み込んで、いま validation data として扱っている情報だけを抜き出す 
            # - memory 圧縮のため id_map <-> series_id の変換が途中で必要
            solution = pd.read_csv('/ebs03/child-mind-institute-detect-sleep-states/data/train_events.csv')
            df_id_map = pd.read_parquet('/ebs03/child-mind-institute-detect-sleep-states/datasets/detect-sleep-states-memory-decrease/train_id_map.parquet')
            valid_id_list = df_series[df_series["fold"]==idx_fold]["id_map"].values
            valid_series_list = df_id_map[df_id_map["id_map"].isin(valid_id_list)]["series_id"].values
            solution = solution[solution["series_id"].isin(valid_series_list)] # valid の該当イベントだけ抽出

            model.eval()
            submission, valid_loss_list = get_submission_events(valid_dataset, model, valid_series_list, DEVICE, mode="valid", criterion=criterion)
            for value in valid_loss_list:
                mlflow_helper.running_update("valid_loss", value)

            pred_score = score(
                solution,
                submission,
                tolerances,
                **column_names
            )
            mlflow_helper.log_metric_epoch(f"{idx_fold}_score", pred_score, epoch=idx_epoch)
            # -----------------------------

            mlflow_helper.batch_update_all()
            mlflow_helper.print_metric(["train_loss", "valid_loss"])

            weight_path = f"/tmp/v4_rnn_{idx_fold}_{idx_epoch}.pth"
            torch.save(model.state_dict(), weight_path)
            mlflow_helper.log_artifact(weight_path)
            mlflow_helper.save_metric(idx_fold, idx_epoch)
        
        pred_score_all.append(pred_score)
    

    mlflow_helper.log_metric("pred_score", np.mean(pred_score_all))
    mlflow_helper.set_terminated()

if __name__ == "__main__" :
    main()