# -*- coding: utf-8 -*-
import threading
from collections import deque
import time
from pylsl import StreamInlet, resolve_byprop
import numpy as np
import os
import itertools
import sys
import keyboard
import joblib

sys.path.insert(1, '../utilities')
from utilities import custom_transform

# cat--
# fft_acc_0.761_recall_0.6433_0.8783_no none
# fft_acc_0.449_recall_0.471_0.4877_0.3478
# fft_acc_0.7166_recall_0.632_0.8014_no none

# Knn
# fft_acc_0.8906_recall_1.0_0.7815_no none

# qda
# fft_acc_0.7246_recall_0.6531_0.7958_no none


TRAINED_MODEL_PATH = '../../models/qda_models/fft_acc_0.7246_recall_0.6531_0.7958_no none'
ACTION = 'right'


class MainProgram(threading.Thread):

    def __init__(self, threadID, name, buff_size):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.data_buffer = deque(maxlen=buff_size)  # define a ring buffer
        self.model = joblib.load(TRAINED_MODEL_PATH + '/model.joblib')
        self.scaler = joblib.load(TRAINED_MODEL_PATH + '/scaler.joblib')
        self.selector = joblib.load(TRAINED_MODEL_PATH + '/selector.joblib')
        self.is_running = True
        self.record_buff = []

    def run(self):
        print('thread started', 'search for pipe ...')
        streams = resolve_byprop('type', 'EEG')
        inlet = StreamInlet(streams[0])
        print('pipe started.')
        while True:
            sample, timestamp = inlet.pull_sample()  # 8 channel for once.
            self.data_buffer.append(sample)
            self.record_buff.append(sample)
            if self.is_running == False:
                break

    def extract_predict(self, sample_size, duration):

        count = 0
        acc_outs = 0
        left = 0
        right = 0
        none = 0
        start = time.time()

        while time.time() < start + duration:

            # check the buffer is filled
            if len(self.data_buffer) < 500:
                continue

            # extract the latest data chunk from buffer [accroding to sample size]
            data_point = np.array(list(itertools.islice(self.data_buffer, 0, sample_size)))  # 8 x 37  array

            # array for keep fft data
            fft_data = np.empty((0, 37))

            # converting and stacking data.
            for i in range(8):
                channel_fft = custom_transform.perform_FFT(data_point[:, i],
                                                           sample_size,
                                                           pass_high=15, pass_low=50,
                                                           zero_filler=False)

                fft_data = np.vstack([fft_data, channel_fft])

            fft_data = np.reshape(fft_data, (1, -1))

            scaled_data = self.scaler.transform(fft_data)
            selected_data = self.selector.transform(scaled_data)

            predicted_class = self.model.predict(selected_data)

            # simulate the keyboard for unity program.
            if predicted_class == 0:
                keyboard.press_and_release('left')
                acc_outs += 1
                left += 1
            elif predicted_class == 1:
                keyboard.press_and_release('right')
                right += 1
            else:
                keyboard.press_and_release('up')
                none += 1

            # count number of outputs
            count += 1

        print('total prediction count : ', count)
        print('accurately predicted ( count / percentage ) : ', acc_outs, round(acc_outs / count))
        print('outper second : ', round(count / duration))
        print('left: ', left, ' right : ', right, ' none ', none)

        np_all = np.array(self.record_buff)  # SCALE_FACTOR_EEG_X24/SCALE_FACTOR_EEG_X1

        # make data directory
        datadir = "../dataset/lsl_data"
        if not os.path.exists(datadir):
            os.mkdir(datadir)

        #make directory for current recording action
        actiondir = f"{datadir}/{ACTION}"
        if not os.path.exists(actiondir):
            os.mkdir(actiondir)

        # print(len(channel_datas))
        print(f"saving {ACTION} data...")
        file_name = f"{int(time.time())}or_{ACTION}.npy"
        np.save(os.path.join(actiondir, file_name), np_all)
        print("done.")


# main execution
keyboard.wait('-')
custom_thread = MainProgram(1, "Thread-1", buff_size=500)
custom_thread.setDaemon(True)
custom_thread.start()
custom_thread.extract_predict(sample_size=256, duration=30)
sys.exit()
os._exit(1)
