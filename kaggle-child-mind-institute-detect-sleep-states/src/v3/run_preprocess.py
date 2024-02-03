import pandas as pd
import numpy as np

# メモリ使用量削減したバージョンの作成

if __name__ == "__main__" :

    # ---- train_events.csv
    train_events = pd.read_csv("/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv")

    new_events = train_events.copy()

    # 277 ids (same id's as in series)
    # new_events.series_id = new_events.series_id.str.filter_characters({'0':'9'}).astype(np.int64)
    # or map it :)
    train_id_map = pd.DataFrame({
        "series_id": new_events.series_id.unique(),
        "id_map": new_events.series_id.unique().index})
    train_id_map["id_map"] = train_id_map["id_map"].astype(np.uint16)
    new_events = pd.merge(new_events, train_id_map, on="series_id").drop(columns="series_id")

    # night
    new_events.night = new_events.night.astype(np.uint16)
    # event relabeled
    new_events.event = new_events.event.replace({'onset':'1', 'wakeup':'2'}).astype(np.uint8)
    # step
    new_events.step = new_events.step.astype(np.uint32)
    # timestamp
    new_events.timestamp = pd.to_datetime(new_events.timestamp, format='%Y-%m-%d %H:%M:%S')
    
    train_id_map.to_parquet("./train_id_map.parquet", index=False)
    new_events.tp_parquet("./train_events.parquet", index=False)


    # ---- train_series.csv
    train_series = pd.read_parquet("/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet")