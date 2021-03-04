import os
import time

import matplotlib.pyplot as plt
import numpy as np
from pylsl import StreamInlet, resolve_byprop

ACTION = 'none'
PASS_BAND_LOW = 10
PASS_BAND_HIGH = 48
LOOP_TIME = 10  # seconds
SCALE_FACTOR_EEG_X24 = 4500000 / 24 / (2 ** 23 - 1)  # uV/count
SAMPLE_RATE = 250.0
SAMPLE_TIME = 1 / SAMPLE_RATE


def startRecordingSample1():
    time.sleep(4)
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    all_data = []
    start = time.time()
    numSamples = 0

    while time.time() < start + LOOP_TIME:
        sample, timestamp = inlet.pull_sample()  # 8 channel for once.
        all_data.append(sample)
        numSamples += 1

    np_all = np.array(all_data)

    # sample plot
    channel_mat = np_all[0:250, 0]
    plt.plot(channel_mat)
    print(np_all.shape)

    # make data directory
    datadir = "../../dataset/lsl_data"
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    # make directory for current recording action
    actiondir = f"{datadir}/{ACTION}"
    if not os.path.exists(actiondir):
        os.mkdir(actiondir)

    print(f"saving {ACTION} data...")
    file_name = f"{int(time.time())}_{ACTION}.npy"
    np.save(os.path.join(actiondir, file_name), np.array(all_data))
    print("done.")


startRecordingSample1()
