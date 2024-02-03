
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime 
from tqdm import tqdm
from copy import deepcopy


class Dataset:
    def __init__(self, path, mode):
        print(">>>> load data")

        self.df_series = pd.read_parquet(f"{path}/{mode}_series.parquet")
        self.mode = mode
        if self.mode == "train":
            # self.df_events = pd.read_csv(f"{path}/{mode}_events.csv")
            self.df_events = pd.read_parquet(f"{path}/{mode}_events.parquet")

        print(">>>> done")

    def make_series_features(self, series_id):

        # target
        tmp_df = deepcopy(self.df_series[self.df_series["series_id"]==series_id])

        # cleaning
        tmp_df = tmp_df.dropna(axis=0, ignore_index=True)
        if self.mode == "trian":
            # dropna によって night (ペア番号) が片方なくなるような行が生じてしまうため、該当のペアも削除する
            event_count = self.df_events[["series_id", "night", "event"]].groupby(by=["series_id", "night"]).count()
            event_count.reset_index(level=[0,1], inplace=True)
            for series_id, night in zip(event_count[event_count["event"] != 2]["series_id"], event_count[event_count["event"] != 2]["night"]):
                self.df_events = self.df_events[~( (self.df_events["series_id"]==series_id) & (self.df_events["night"]==night) )].reset_index(drop=True)

        # preprocess
        # (1) series
        tmp_df["timestamp"] = pd.to_datetime(tmp_df['timestamp']).apply(lambda t: t.tz_localize(None))

        # datetime
        tmp_df.loc[:, "year"] = tmp_df["timestamp"].dt.year
        tmp_df.loc[:, "month"] = tmp_df["timestamp"].dt.month
        tmp_df.loc[:, "day"] = tmp_df["timestamp"].dt.day
        tmp_df.loc[:, "hour"] = tmp_df["timestamp"].dt.hour
        tmp_df["dayofweek"] = tmp_df["timestamp"].dt.dayofweek
        tmp_df["weekend"] = tmp_df["timestamp"].dt.dayofweek > 4
        ## 四季
        month_to_season = dict(zip(range(1,13), [month % 12// 3 + 1 for month in range(1, 13)]))
        tmp_df["season"] = tmp_df["timestamp"].dt.month.map(month_to_season)

        tmp_df.loc[:, "anglez"] = abs(tmp_df["anglez"])

        # diff
        periods = 20
        tmp_df.loc[:, "anglez_diff"] = tmp_df.groupby('series_id')['anglez'].diff(periods=periods).bfill().astype('float16')
        tmp_df.loc[:, "enmo_diff"] = tmp_df.groupby('series_id')['enmo'].diff(periods=periods).bfill().astype('float16')
        # rolling (移動平均)
        tmp_df.loc[:, "anglez_rolling_mean"] = tmp_df["anglez"].rolling(periods,center=True).mean().bfill().ffill().astype('float16')
        tmp_df.loc[:, "enmo_rolling_mean"] = tmp_df["enmo"].rolling(periods,center=True).mean().bfill().ffill().astype('float16')
        tmp_df.loc[:, "anglez_rolling_max"] = tmp_df["anglez"].rolling(periods,center=True).max().bfill().ffill().astype('float16')
        tmp_df.loc[:, "enmo_rolling_max"] = tmp_df["enmo"].rolling(periods,center=True).max().bfill().ffill().astype('float16')
        tmp_df.loc[:, "anglez_rolling_std"] = tmp_df["anglez"].rolling(periods,center=True).std().bfill().ffill().astype('float16')
        tmp_df.loc[:, "enmo_rolling_std"] = tmp_df["enmo"].rolling(periods,center=True).std().bfill().ffill().astype('float16')
        tmp_df.loc[:, "anglez_diff_rolling_mean"] = tmp_df["anglez_diff"].rolling(periods,center=True).mean().bfill().ffill().astype('float16')
        tmp_df.loc[:, "enmo_diff_rolling_mean"] = tmp_df["enmo_diff"].rolling(periods,center=True).mean().bfill().ffill().astype('float16')
        tmp_df.loc[:, "anglez_diff_rolling_max"] = tmp_df["anglez_diff"].rolling(periods,center=True).max().bfill().ffill().astype('float16')
        tmp_df.loc[:, "enmo_diff_rolling_max"] = tmp_df["enmo_diff"].rolling(periods,center=True).max().bfill().ffill().astype('float16')
        
        # (2) events
        if self.mode == "train" :
            tmp_events = deepcopy(self.df_events[self.df_events["series_id"]==series_id])
            tmp_events["step"] = tmp_events["step"] 
            tmp_events["awake"] = tmp_events["event"].replace({"onset":1, "wakeup":0}) # 二値分類

            df_data = pd.merge(tmp_df, tmp_events[["step", "awake"]], on="step", how="left")
            df_data["awake"] = df_data["awake"].bfill().ffill().fillna(1).astype("int") # TBD: fillna ?
            # [nan, nan, ..., 1, nan, nan, ..., 0, nan, nan, ...] という状態にして
            # bfill (直後の値で fill)、ffill（直前の値でfill）という形で、onset <--> wakeup 間を埋める

            return df_data 
        else:
            return tmp_df


def get_events(df):
    '''
    Takes a time series and a classifier and returns a formatted submission dataframe.
    '''
    
    series_ids = df['series_id'].unique()
    events = {
        "series_id" : [],
        "step" : [],
        "event" : [],
        "score" : [],
    }

    for idx in tqdm(series_ids) : 
        
        # target
        tmp_df = df[df["series_id"]==idx] 

        # Getting predicted onset and wakeup time steps
        pred_onsets = tmp_df[tmp_df['pred'].diff() > 0]['step'].values 
        pred_wakeups = tmp_df[tmp_df['pred'].diff() < 0]['step'].values
        
        if len(pred_onsets) > 0 : 
            
            # Ensuring all predicted sleep periods begin and end
            if min(pred_wakeups) < min(pred_onsets) : 
                pred_wakeups = pred_wakeups[1:]

            if max(pred_onsets) > max(pred_wakeups) :
                pred_onsets = pred_onsets[:-1]

            # Keeping sleep periods longer than 30 minutes
            # 5 sec/ step なので、5 * 12 -> 1 min
            sleep_periods = [(onset, wakeup) for onset, wakeup in zip(pred_onsets, pred_wakeups) if (wakeup - onset) >= 12 * 30]

            for onset, wakeup in sleep_periods :
                # Scoring using mean probability over period
                score = tmp_df[(tmp_df['step'] >= onset) & (tmp_df['step'] <= wakeup)]['prob'].mean()

                events["series_id"].append(idx)
                events["step"].append(onset)
                events["event"].append("onset")
                events["score"].append(score)
                
                events["series_id"].append(idx)
                events["step"].append(wakeup)
                events["event"].append("wakeup")
                events["score"].append(score)

    # Adding row id column
    events = pd.DataFrame(events)
    events = events.reset_index().rename(columns={"index":"row_id"})
    return events