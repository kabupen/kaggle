import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

# メモリ使用量削減したバージョンの作成

if __name__ == "__main__" :

    ## # ---- train_events.csv
    ## train_series = pd.read_parquet("/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet")
    ## train_events = pd.read_csv("/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv")

    ## new_series = train_series.copy()
    ## df_new_events = train_events.copy()

    ## # 277 ids (same id's as in series)
    ## # df_new_events.series_id = df_new_events.series_id.str.filter_characters({'0':'9'}).astype(np.int64)
    ## # or map it :)
    ## train_id_map = pd.DataFrame({
    ##     "series_id": df_new_events.series_id.unique(),
    ##     "id_map": df_new_events.series_id.unique().index})
    ## train_id_map["id_map"] = train_id_map["id_map"].astype(np.uint16)
    ## df_new_events = pd.merge(df_new_events, train_id_map, on="series_id").drop(columns="series_id")

    ## # night
    ## df_new_events.night = df_new_events.night.astype(np.uint16)
    ## # event relabeled
    ## df_new_events.event = df_new_events.event.replace({'onset':'1', 'wakeup':'2'}).astype(np.uint8)
    ## # step
    ## df_new_events.step = df_new_events.step.astype(np.uint32)
    ## # timestamp
    ## df_new_events.timestamp = pd.to_datetime(df_new_events.timestamp, format='%Y-%m-%d %H:%M:%S')
    
    ## train_id_map.to_parquet("./train_id_map.parquet", index=False)
    ## df_new_events.tp_parquet("./train_events.parquet", index=False)


    ## # ---- train_series.csv
    ## train_id_map = pd.read_parquet("/kaggle/input/detect-sleep-states-memory-decrease/train_id_map.parquet")
    ## df_new_series = train_series.merge(right=train_id_map, on="series_id")
    ## df_new_series = df_new_series.drop(columns="series_id").reset_index(drop=True)

    ## df_new_series.step = df_new_series.step.astype(np.uint32)

    ## from pandarallel import pandarallel
    ## pandarallel.initialize(progress_bar=True)

    ## # Local Time converter
    ## def to_date_time(x):
    ##     import pandas as pd
    ##     return pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S') # utc=True

    ## def to_localize(t):
    ##     import pandas as pd
    ##     return t.tz_localize(None)

    ## df_new_series["timestamp"] = df_new_series.timestamp.parallel_apply(to_date_time).parallel_apply(to_localize)


    ts = pd.read_parquet("/ebs03/child-mind-institute-detect-sleep-states/datasets/detect-sleep-states-memory-decrease/train_series.parquet")
    te = pd.read_parquet("/ebs03/child-mind-institute-detect-sleep-states/datasets/detect-sleep-states-memory-decrease/train_events.parquet")

    # left merge なので該当しない行は NaN で結合される（=eventが発生していない時刻）
    tse = pd.merge(ts, te[["id_map", "step", "event"]], on=["id_map", "step"], how="left")
    tse["event"] = tse["event"].fillna(0).astype("uint8")
    
    # one-hot 
    tse = pd.get_dummies(tse, columns=["event"], dtype="uint8")
    print(tse)
    tse = tse.drop(columns={"event_0"})

    # group k-fold
    kf = GroupKFold(n_splits=5)
    for idx_fold, (_, valid_index) in enumerate(kf.split(tse, tse[['event_1', "event_2"]], tse['id_map'])):
        print('FOLD{}'.format(idx_fold))
        tse.loc[valid_index, "fold"] = idx_fold
    
    tse[["id_map", "step", "anglez", "enmo", "fold"]].to_pickle("/ebs03/child-mind-institute-detect-sleep-states/output/features/v4/prep_train_series.pkl")
    tse[["id_map", "step", "event_1", "event_2", "fold"]].to_pickle("/ebs03/child-mind-institute-detect-sleep-states/output/features/v4/prep_train_events.pkl")