import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import os
from tqdm import tqdm
from argparse import ArgumentParser


def add_feature(df):

    def to_sin(x):
        rad = 2 * np.pi * (x % max_) / max
    # datetime
    df.loc[:, "year"] = df["timestamp"].dt.year
    df.loc[:, "month"] = df["timestamp"].dt.month
    df.loc[:, "day"] = df["timestamp"].dt.day
    df.loc[:, "hour"] = df["timestamp"].dt.hour
    df.loc[:, "minute"] = df["timestamp"].dt.minute
    # tmp_df["dayofweek"] = tmp_df["timestamp"].dt.dayofweek
    # tmp_df["weekend"] = tmp_df["timestamp"].dt.dayofweek > 4

    df.loc[:, "month_sin"] = df.loc[:, "month"].apply(lambda x : np.sin( (2* np.pi* (x%12)) / 12 ))
    df.loc[:, "hour_sin"] = df.loc[:, "hour"].apply(lambda x : np.sin( (2* np.pi* (x%24)) / 24 ))
    df.loc[:, "minute_sin"] = df.loc[:, "minute"].apply(lambda x : np.sin( (2* np.pi* (x%60)) / 60 ))
    df.loc[:, "month_cos"] = df.loc[:, "month"].apply(lambda x : np.cos( (2* np.pi* (x%12)) / 12 ))
    df.loc[:, "hour_cos"] = df.loc[:, "hour"].apply(lambda x : np.cos( (2* np.pi* (x%24)) / 24 ))
    df.loc[:, "minute_cos"] = df.loc[:, "minute"].apply(lambda x : np.cos( (2* np.pi* (x%60)) / 60 ))

    # others
    n_step = np.max(df["step"].dropna())
    df.loc[:, "step"] = df["step"].apply(lambda x : x/n_step)
    df.loc[:, "anglez_sin"] = df["anglez_rad"].apply(lambda x : np.sin(x))
    df.loc[:, "anglez_cos"] = df["anglez_rad"].apply(lambda x : np.cos(x))

    return df

def save_each_series(output_dir, df):
    os.makedirs(output_dir, exist_ok=True)

    for col_name in list(df.columns):
        feat = df[col_name].values
        np.save(f"{output_dir}/{col_name}", feat)

def main(mode):

    ANGLEZ_MEAN = -8.810476
    ANGLEZ_STD = 35.521877
    ENMO_MEAN = 0.041315
    ENMO_STD = 0.101829

    if mode == "local":
        df_series = pd.read_parquet("/ebs03/child-mind-institute-detect-sleep-states/data/train_series.parquet")
        OUTPUT_DIR = "/ebs03/child-mind-institute-detect-sleep-states/output/features/v5/train"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    elif mode == "submit" :
        df_series = pd.read_parquet("/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet")
        OUTPUT_DIR = "/kaggle/working/submit"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # preprocess
    df_series["timestamp"] = pd.to_datetime(df_series["timestamp"]).apply(lambda t: t.tz_localize(None))
    df_series["anglez_rad"] = np.deg2rad(df_series["anglez"])
    df_series["anglez"] = df_series["anglez"].apply(lambda x : (x - ANGLEZ_MEAN)/ANGLEZ_STD)
    df_series["enmo"] = df_series["enmo"].apply(lambda x : (x - ENMO_MEAN)/ENMO_STD)

    df_series = df_series.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    n_unique = len(df_series["series_id"].unique())
    for series_id, this_df in tqdm(df_series.groupby("series_id"), total=n_unique):
        this_df = add_feature(this_df)
        output_dir = f"{OUTPUT_DIR}/{series_id}"
        save_each_series(output_dir, this_df)


if __name__ == "__main__" :
    argparser = ArgumentParser()
    argparser.add_argument('-m', '--mode', type=str, choices=["local", "submit"], default="local")
    args = argparser.parse_args()
    main(args.mode)