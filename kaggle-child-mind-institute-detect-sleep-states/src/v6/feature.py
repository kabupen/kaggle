
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime 
from tqdm import tqdm
from copy import deepcopy


class Dataset:
    def __init__(self, path, mode):
        print(">>>> load data")

        self.df_full_series = pd.read_parquet(f"{path}/{mode}_series.parquet")
        self.mode = mode
        self.id_key = "series_id"

        # ローカル環境 
        if self.mode == "train":
            self.df_full_events = pd.read_parquet(f"{path}/{mode}_events.parquet")
            self.id_key = "id_map"

        print(">>>> done")

    def make_series_features(self, series_id):

        # target
        df_series = deepcopy(self.df_full_series[self.df_full_series[self.id_key]==series_id])

        # cleaning
        df_series = df_series.dropna(axis=0, ignore_index=True)
        if self.mode == "trian":
            # dropna によって night (ペア番号) が片方なくなるような行が生じてしまうため、該当のペアも削除する
            event_count = self.df_full_events[[self.id_key, "night", "event"]].groupby(by=[self.id_key, "night"]).count()
            event_count.reset_index(level=[0,1], inplace=True)
            for series_id, night in zip(event_count[event_count["event"] != 2][self.id_key], event_count[event_count["event"] != 2]["night"]):
                self.df_full_events = self.df_full_events[~( (self.df_full_events[self.id_key]==series_id) & (self.df_full_events["night"]==night) )].reset_index(drop=True)

        # preprocess
        # (1) series
        df_series["timestamp"] = pd.to_datetime(df_series['timestamp']).apply(lambda t: t.tz_localize(None))

        # datetime
        df_series.loc[:, "year"] = df_series["timestamp"].dt.year
        df_series.loc[:, "month"] = df_series["timestamp"].dt.month
        df_series.loc[:, "day"] = df_series["timestamp"].dt.day
        df_series.loc[:, "hour"] = df_series["timestamp"].dt.hour
        df_series["dayofweek"] = df_series["timestamp"].dt.dayofweek
        df_series["weekend"] = df_series["timestamp"].dt.dayofweek > 4
        ## 四季 (1,2,3,4)
        month_to_season = dict(zip(range(1,13), [month % 12// 3 + 1 for month in range(1, 13)]))
        df_series["season"] = df_series["timestamp"].dt.month.map(month_to_season)

        df_series.loc[:, "anglez"] = abs(df_series["anglez"])

        for mins in [5, 30, 60*2, 60*8]: # [分]
            for var in ["enmo", "anglez"]:
                # diff
                df_series.loc[:, f"{var}_diff"] = df_series.groupby(self.id_key)[var].diff(periods=mins).bfill().astype('float16')

                # rolling
                df_series.loc[:, f"{var}_rolling_mean"] = df_series[var].rolling(mins * 12, center=True).mean().bfill().ffill().astype('float16')
                df_series.loc[:, f"{var}_rolling_max"] = df_series[var].rolling(mins * 12, center=True).max().bfill().ffill().astype('float16')
                df_series.loc[:, f"{var}_rolling_std"] = df_series[var].rolling(mins * 12, center=True).std().bfill().ffill().astype('float16')
                
                # diff
                df_series.loc[:, f"{var}_diff_rolling_mean"] = df_series[f"{var}_diff"].rolling(mins*12, center=True).mean().bfill().ffill().astype('float16')
                df_series.loc[:, f"{var}_diff_rolling_std"] = df_series[f"{var}_diff"].rolling(mins*12, center=True).std().bfill().ffill().astype('float16')
                df_series.loc[:, f"{var}_diff_rolling_max"] = df_series[f"{var}_diff"].rolling(mins*12, center=True).max().bfill().ffill().astype('float16')
        
        if self.mode == "train" :
            # (2) events (ground truth)
            df_events = deepcopy(self.df_full_events[self.df_full_events[self.id_key]==series_id])

            # 正解ラベルについて
            # (1) 二値分類
            # df_events["awake"] = df_events["event"].replace({"onset":1, "wakeup":0})
            # df_data = pd.merge(df_series, df_events[["step", "awake"]], on="step", how="left")
            # df_data["awake"] = df_data["awake"].bfill().ffill().fillna(0).astype("int") # TBD: fillna ?
            # [nan, nan, ..., 1, nan, nan, ..., 0, nan, nan, ...] という状態にして
            # bfill (直後の値で fill)、ffill（直前の値でfill）という形で、onset <--> wakeup 間を埋める

            # (2) 多クラス分類
            # df_events["awake"] = df_events["event"].replace({"onset":1, "wakeup":2}) 
            df_data = pd.merge(df_series, df_events[["step", "event"]], on="step", how="left")
            df_data["event"] = df_data["event"].fillna(0)

            return df_data 
        else:
            return df_series


def get_events(df, id_key="series_id"):
    '''
    Takes a time series and a classifier and returns a formatted submission dataframe.
    '''
    
    series_ids = df[id_key].unique()
    events = {
        id_key : [],
        "step" : [],
        "event" : [],
        "score" : [],
    }

    for idx in tqdm(series_ids) : 
        
        # target
        tmp_df = df[df[id_key]==idx] 

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

                events[id_key].append(idx)
                events["step"].append(onset)
                events["event"].append("onset")
                events["score"].append(score)
                
                events[id_key].append(idx)
                events["step"].append(wakeup)
                events["event"].append("wakeup")
                events["score"].append(score)

    # Adding row id column
    events = pd.DataFrame(events)
    events = events.reset_index().rename(columns={"index":"row_id"})
    return events