# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 01:37:16 2020
before use this assign the direction going to use
@author: Avishka Dhananjaya Athapattu
"""

from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque
import os

#predefined variables
ACTION = 'left' # current action
fps_counter = deque(maxlen=150)

print("looking for an EEG stream...")       # first resolve an EEG stream on the lab network
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])             # create a new inlet to read from the stream

channel_datas = []
start_time = time.time()
last_update = time.time()

while time.time() < start_time+5:  # how many iterations. Eventually this would be a while True    
    sample, timestamp = inlet.pull_sample()        
    fps_counter.append(time.time() - last_update)
    last_update = time.time()
    cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    print(cur_raw_hz)
    
    channel_datas.append(sample)

# print(np.array(channel_datas).shape)

# # make data directory
# datadir = "../../dataset/raw_data"
# if not os.path.exists(datadir):
#     os.mkdir(datadir)
    
# # make directory for current recording action
# actiondir = f"{datadir}/{ACTION}"
# if not os.path.exists(actiondir):
#     os.mkdir(actiondir)

# print(len(channel_datas))
# print(f"saving {ACTION} data...")
# np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
# print("done.")



