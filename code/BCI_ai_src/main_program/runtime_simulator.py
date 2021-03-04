# -*- coding: utf-8 -*-
import threading
from collections import deque
import time
import numpy as np
import os
import itertools
import sys
import keyboard
import datetime
import joblib

sys.path.insert(1, '../utilities')
from utilities import custom_transform

TRAINED_MODEL_PATH = '../../models/cat_models/fft_acc_0.761_recall_0.6433_0.8783_no none/'
SIMULATE_DIR = '../dataset/simulation_lsl_data/'  # simulation_


class MainProgram(threading.Thread):

    def __init__(self, threadID, name, buff_size):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.data_buffer = deque(maxlen=buff_size)  # define a ring buffer
        self.model = joblib.load(TRAINED_MODEL_PATH + 'model.joblib')
        self.scaler = joblib.load(TRAINED_MODEL_PATH + 'scaler.joblib')
        self.selector = joblib.load(TRAINED_MODEL_PATH + 'selector.joblib')
        self.is_running = True

    def run(self):
        print('thread started\nsearch for pipe ...')
        for root, dirs, files in os.walk(SIMULATE_DIR):
            if len(files) == 0:  # if directory empty skip that
                continue

            dir_name = root.split('/')[-1]  # get the directory name of the direction
            print(dir_name)

            if dir_name != 'left':
                continue

            assesed_files = 0  # currently accessed files

            for file in files:
                if self.is_running == False:
                    break
                if file.endswith(".npy"):
                    print(file)
                    dataset = np.load(root + "/" + file)  # load files.
                    dataset_len = dataset.shape[0]  # dataset length (approximately 2500) do not depend on this.

                    for init_index in range(dataset_len):  # iterate through dataset while extracting 256 data points
                        self.data_buffer.append(dataset[init_index, :])
                        import time
                        time.sleep(0.004)

    def extract_predict(self, sample_size, duration):

        count = 0
        acc_outs = 0
        left = 0
        right = 0
        none = 0
        start = time.time()

        while time.time() < start + duration:

            # print(len(self.data_buffer))

            # check the buffer is filled
            if (len(self.data_buffer) < 500):
                continue

            # extract the latest data chunk from buffer [accroding to sample size]
            data_point = np.array(list(itertools.islice(self.data_buffer, 0, sample_size)))  # 8 x 37  array

            # array for keep fft data
            fft_data = np.empty((0, 37))

            # converting and stacking data.
            for i in range(8):
                channel_ts = data_point[:, i]
                channel_fft = custom_transform.perform_FFT(channel_ts, sample_size,
                                                           pass_high=15, pass_low=50,
                                                           zero_filler=False)
                fft_data = np.vstack([fft_data, channel_fft])

            fft_data = np.reshape(fft_data, (1, -1))  # reshape into 1 x 296 array

            scaled_data = self.scaler.transform(fft_data)  # normalize the dataset
            selected_data = self.selector.transform(scaled_data)  # feature selections.

            start_t = datetime.datetime.now()
            predicted_class = self.model.predict(selected_data)  # predict the data.
            end_t = datetime.datetime.now()
            elapsed_time = end_t - start_t
            # print('elapsed time per instance ' , elapsed_time.microseconds)

            # simulate the keyboard for unity program.
            if predicted_class == 0:
                # keyboard.press_and_release('left')
                left += 1
            elif predicted_class == 1:
                # keyboard.press_and_release('right')
                right += 1
            elif predicted_class == 4:
                # keyboard.press_and_release('up')
                none += 1

            # count number of outputs
            count += 1

        print('total prediction count : ', count)
        print('left: ', left, ' right : ', right, ' none : ', none)


# main execution
keyboard.wait('-')
custom_thread = MainProgram(1, "Thread-1", buff_size=500)
custom_thread.setDaemon(True)
custom_thread.start()
custom_thread.extract_predict(sample_size=256, duration=80)
custom_thread.is_running = False
sys.exit()
