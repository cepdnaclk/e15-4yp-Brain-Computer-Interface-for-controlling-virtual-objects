# -*- coding: utf-8 -*-
"""

Created on Thu Oct 15 15:27:56 2020
serial connection: com11
no lsl

fps: (bursting)550 - 600 for cyton
fps: (normal)120 -130
fps: daisy unknown

@author: Avishka Dhananjaya Athapattu.

"""
from pyOpenBCI import OpenBCICyton
import time
from collections import deque
import numpy as np

DATA_ROWS = 100
CYTON_CHANNELS = 8
DAISY_CHANNELS = 16

last_print = time.time();
fps_counter = deque(maxlen=50)
sequence = np.zeros(shape=(DATA_ROWS, CYTON_CHANNELS))
counter = 0


def print_raw(sample):
    global last_print
    global sequence
    global counter

    sequence = np.roll(sequence, shift=1, axis=0)
    sequence[0, ...] = sample.channels_data

    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    print(f'FPS:{1 / (sum(fps_counter) / len(fps_counter)):.2f},:{len(sequence)},...{counter}')

    # counter+=1
    # if counter == 30000:
    #     np.save(f"seq.npy",sequence)
    #     break


board = OpenBCICyton(port='COM11', daisy=False)
board.start_stream(print_raw)
