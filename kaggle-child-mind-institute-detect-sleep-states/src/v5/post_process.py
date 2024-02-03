
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def post_process( keys: list[str], preds: np.ndarray, score_th: float = 0.01, distance: int = 5000) -> pd.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """

    # key : {series_id}_{index}
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        # 該当の series id だけ抽出
        #  - [#chunk, #時系列, 2] になっているので reshape して [#chunk*#時系列, 2] にする
        #  - #chunk * #時系列 は実際の series_id の全時系列情報
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)

        for idx, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, idx] # 該当の event の 横軸:全時系列情報[t], 縦軸:y_pred の情報を取得
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0] # (np.array, {'peak_heights':np.array()}})
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    df_sub = pd.DataFrame(records).sort_values(["series_id", "step"]).reset_index(drop=True)
    return df_sub