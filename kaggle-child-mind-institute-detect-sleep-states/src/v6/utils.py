import numpy as np
import random

def random_crop(pos, duration, max_end):
    """
    pos を含む duration ウィンドウサイズのインデックスを計算する
    """
    start = random.randint(max(0, pos-duration), min(pos, max_end-duration))
    end = start + duration
    return start, end