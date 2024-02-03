import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from copy import deepcopy
import hydra
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler

import sys
sys.path.append("/workspace/child-mind-institute-detect-sleep-states/src/")
from common.mlflow_helper import MlflowHelper
from dataset import SleepTrainDataset, SleepValidDataset, SleepTestDataset, load_features, load_chunk_features
from model.common import get_model
from metrics import event_detection_ap, score
from post_process import post_process

def run_infer(series_id_list, cfg, model, device):
    model.eval()
    df_submit = pd.DataFrame()
    for series_id in tqdm(series_id_list):
        # test loader
        chunk_features = load_chunk_features(
            data_path=cfg.feature_dir,
            n_frames=cfg.n_frames,
            feature_names=cfg.feature_names,
            series_id_list=[series_id,]
        )
        test_dataset = SleepTestDataset(chunk_features, cfg.n_frames)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        preds = []
        keys = []
        for batch in test_loader:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    output = model(batch["X"].to(device)) # [bs, #時系列長, #y]
                    y_pred = torch.sigmoid(output)[:, :, [1, 2]] # onset/wakeup だけ抽出
                
                preds.append(y_pred.detach().cpu().numpy())
                keys.extend(batch["key"])

        preds = np.concatenate(preds)

        # make submission.csv
        tmp_sub = post_process(
            keys,
            preds,  
            score_th=cfg.post_process.score_th,
            distance=cfg.post_process.distance, 
        )
        df_submit = pd.concat([df_submit, tmp_sub], axis=0)
    return df_submit

@hydra.main(config_path="../../config", config_name="")
def main(cfg):    
    
    DEVICE = "cuda"

    # mlflow
    mlflow_helper = MlflowHelper(cfg.mlflow_output_path, cfg.experiment_name)
    mlflow_helper.create_run()

    # log parameters
    mlflow_helper.log_params(cfg)

    df_events = pd.read_csv("/ebs03/child-mind-institute-detect-sleep-states/data/train_events.csv")
    with open("/ebs03/child-mind-institute-detect-sleep-states/output/v5/kfold/kfold_series_id.pkl", "rb") as f:
        series_id_dict = pickle.load(f)

    pred_score_all = []
    for idx_fold in range(cfg.num_fold):
        train_series_id = series_id_dict[f"f{idx_fold}"]["train"]
        valid_series_id = series_id_dict[f"f{idx_fold}"]["valid"]

        model = get_model(cfg).to(DEVICE)

        train_features = load_features(
            "/ebs03/child-mind-institute-detect-sleep-states/output/v5/features/", 
            train_series_id, 
            cfg.feature_names
        )
        valid_features = load_chunk_features(
            "/ebs03/child-mind-institute-detect-sleep-states/output/v5/features/", 
            cfg.n_frames, 
            valid_series_id, 
            cfg.feature_names
        )
        df_train_events = df_events[df_events["series_id"].isin(train_series_id)]
        df_valid_events = df_events[df_events["series_id"].isin(valid_series_id)]
        train_dataset = SleepTrainDataset(train_features, df_train_events, n_frames=cfg.n_frames, cfg=cfg)
        valid_dataset = SleepValidDataset(valid_features, df_valid_events, n_frames=cfg.n_frames)
        train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, pin_memory=True, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.valid_batch_size, pin_memory=True, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        # steps = int(len(train_dataset)*0.9)*5
        # scheduler = CosineLRScheduler(optimizer, t_initial=steps, warmup_t=int(steps*0.2), warmup_lr_init=1e-6, lr_min=2e-8,)
        if cfg.criterion == "BCEWithLogitsLoss":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif cfg.criterion == "CrossEntropyLoss": 
            criterion = torch.nn.CrossEntropyLoss()
        else :
            raise ValueError(cfg.criterion)

        for idx_epoch in range(cfg.num_epoch):

            ### train
            model.train()
            for idx_step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

                optimizer.zero_grad()
                # scheduler.step(idx_step)

                # forward
                do_mixup = np.random.rand() < cfg.aug.mixup_prob
                do_cutmix = np.random.rand() < cfg.aug.cutmix_prob
                # input: [bs, #時系列, #特徴量]
                # output: [bs, #時系列, 3] (awake, onset, wakeup)
                output = model(batch["X"].to(DEVICE), batch["y_true"].to(DEVICE), do_mixup, do_cutmix)
                gc.collect()

                loss = criterion(output.float().cpu(), batch["y_true"].float().cpu())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optimizer.grad_norm)
                optimizer.step()

                mlflow_helper.running_update(f"train_loss", loss.item())

                del batch 
                gc.collect()

            ### valid
            # model.eval()
            # validation_output = []
            # for idx_step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            #     with torch.no_grad():
            #         output = model(batch["X"].to(DEVICE))
            #         if cfg.criterion == "BCEWithLogitsLoss":
            #             y_pred = torch.sigmoid(output)[:, :, [1, 2]] # onset/wakeup だけ抽出
            #         elif cfg.criterion == "CrossEntropyLoss": 
            #             y_pred = torch.softmax(output, dim=2)[:, :, [1, 2]] # onset/wakeup だけ抽出
            #         else :
            #             raise ValueError(cfg.criterion)
            #         # save info
            #         validation_output.append({
            #             "key": batch["key"],
            #             "y_true": batch["y_true"].detach().cpu().numpy(),
            #             "y_pred": y_pred.detach().cpu().numpy(),
            #         })
            #         loss = criterion(output.float().cpu(), batch["y_true"].float().cpu())
            #     
            #     mlflow_helper.running_update(f"valid_loss", loss)

            #     del batch
            #     gc.collect()
            
            # # 集計
            # # key : batch size 分が詰まったリストになっていて、shuffle=False なので {series_id}_{index} が連番で入っている
            # keys = []
            # for x in validation_output:
            #     keys.extend(x["key"])
            # preds = np.concatenate([x["y_pred"] for x in validation_output])

            # df_submit = post_process(
            #     keys=keys,
            #     preds=preds,
            #     score_th=cfg.post_process.score_th,
            #     distance=cfg.post_process.distance,
            # )
            df_submit = run_infer(valid_series_id, cfg, model, DEVICE)

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
            # pred_score = score(
            #     df_valid_events, 
            #     df_submit, # df_pred_events,
            #     tolerances,
            #     **column_names
            # )
            pred_score = event_detection_ap(df_valid_events, df_submit)

            mlflow_helper.update(f"score", pred_score)
            # -----------------------------
            mlflow_helper.batch_update_all()
            # mlflow_helper.print_metric(["train_loss", "valid_loss", "score"])
            mlflow_helper.print_metric(["score"])

            mlflow_helper.save_metric(idx_fold, idx_epoch)

            # weight 
            weight_path = f"/tmp/v4_rnn_{idx_fold}_{idx_epoch}.pth"
            torch.save(model.state_dict(), weight_path)
            mlflow_helper.log_artifact(weight_path)

            gc.collect()

        pred_score_all.append(pred_score)

        # 1 fold のみ 
        if cfg.one_fold : break

    mlflow_helper.log_metric(f"pred_score_all", np.mean(pred_score_all))
    mlflow_helper.set_terminated()

if __name__ == "__main__" :
    main()