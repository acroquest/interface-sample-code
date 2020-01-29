# coding: utf-8

import time

current_time = None


def measure_lap(show=False):
    """
    時間計測。前回呼ばれたときからの経過時間を返す。
    """
    global current_time

    now_time = time.time()
    if current_time is None:
        current_time = now_time

    lap_time = now_time - current_time
    current_time = now_time

    if show:
        print('{:.3f}sec'.format(lap_time))

    return lap_time
