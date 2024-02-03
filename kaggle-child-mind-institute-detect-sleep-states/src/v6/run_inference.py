from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
import hydra
from glob import glob

from model.common import get_model
from dataset import load_chunk_features, SleepTestDataset
from post_process import post_process

@hydra.main(config_path="../../config", config_name="")
def main(cfg):

    # preprocess
    #   convert test_series.parquet into each numpy features

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)

    # load weights
    if cfg.submit.weight is not None:
        model.load_state_dict(torch.load(cfg.submit.weight))
        print('load weight from "{}"'.format(cfg.submit.weight))
    model.eval()

    # per series_id
    df_submit = pd.DataFrame()
    df_series = pd.read_parquet("/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet")
    series_id_list = df_series["series_id"].unique()
    for series_id in series_id_list:
        # test loader
        feature_dir = "/kaggle/working/submit/"
        chunk_features = load_chunk_features(
            data_path=feature_dir,
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
        for batch in tqdm(test_loader, desc="inference"):
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

    # concat しているので index がバラバラになっているため、reset_indexを2回実行する
    df_submit.reset_index(drop=True).reset_index().rename(columns={"index":"row_id"}).to_csv("/kaggle/working/submission.csv", index=False)

if __name__ == "__main__":
    main()