# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 00:27:59 2020
    data_format
        row 0: number of packages.

@author: Prophet X
"""
import time
import numpy as np
import os

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import matplotlib.pyplot as plt

ACTION = 'left'  # Action
SCALE_FACTOR_EEG = (4500000) / 24 / (2 ** 23 - 1)  # uV/count

fs = 250.0  # sample rate
lowcut = 20.0  # low cutoff frequency
highcut = 60.0  # high cutoff frequency
N = 256  # sample count

# cyton board controlling parameters
BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
params.serial_port = 'COM11'
board_id = BoardIds.CYTON_BOARD.value
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)

board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream(45000, 'file://cyton_data.csv:w')
time.sleep(6)

channel_datas = []  # keep channel data points [# of data points * # of channels * # of data points ]

try:
    start_time = time.time()  # start time for accquire data points.

    while time.time() < start_time + 10:  # get data approximately for 10 seconds.
        current_data = board.get_current_board_data(256)  # SCALE_FACTOR_EEG  #take latest 256 data points. ( uV )
        channel_datas.append(current_data[eeg_channels])
        time.sleep(0.045)

    print(np.array(channel_datas).shape)

    # make data directory
    datadir = "../../dataset/raw_data"
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    # make directory for current recording action
    actiondir = f"{datadir}/{ACTION}"
    if not os.path.exists(actiondir):
        os.mkdir(actiondir)

    filename = f"{int(time.time())}_{ACTION}.npy"
    np.save(os.path.join(actiondir, filename), np.array(channel_datas))
    print("done.")

    board.stop_stream()
    board.release_session()

    dataset = np.load(actiondir + "/" + filename)

    sample_data = dataset[5]  # 8 x 256 data points
    plt.figure(1)
    for i in range(8):
        plt.subplot(240 + i + 1)
        plt.plot(sample_data[i])

    plt.figure(2)
    channel_data = sample_data[0]  # channel 1 data
    DataFilter.perform_bandpass(channel_data, int(fs), 40.0, 50, 3, FilterTypes.BESSEL.value, 0)
    DataFilter.perform_bandstop(channel_data, int(fs), 50.0, 2.0, 3, FilterTypes.BESSEL.value, 0)
    plt.plot(channel_data)


except:
    print('exception')
    board.stop_stream()
    board.release_session()
