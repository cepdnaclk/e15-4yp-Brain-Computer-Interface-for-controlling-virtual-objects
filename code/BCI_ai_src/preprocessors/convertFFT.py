import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, '../utilities')
from utilities import custom_transform
from utilities import DATA

print('x')
plt.close('all')

root_raw = '../dataset/lsl_data'
root_transformed = '../dataset/transformed_data'
root_dfft = '../dataset/downloaded_fft_dataset/model_data/data'
root_lsl = '../dataset/lsl_data/'

fs = 250.0  # sample rate Hz
lowcut = 15  # low cutoff frequency
highcut = 50  # high cutoff frequency
N = 256  # number of samples.
step_size = 40  # sampling step size

directory = os.path.join(root_raw)

# create dataframe name
df_columns = ['channel']
for i in range(37):
    frequency_bin = 'f_' + str(i + 1)
    df_columns.append(frequency_bin)
df_columns.append('direction')

for root, dirs, files in os.walk(root_raw):

    if len(files) == 0:  # if directory empty skip that
        continue

    dir_name = root.split('\\')[-1]  # get the directory name of the direction
    print(dir_name)

    if dir_name == 'none':
        continue

    train_2_wr = np.empty((0, 39))  # empty mat for train data
    test_2_wr = np.empty((0, 39))  # empty mat for test data
    max_train_count = int(len(files) * 90 / 100)  # maximum number of training files.
    assesed_files = 0  # currently accessed files

    test_set_files = []

    files.sort(reverse=True)  # arranging the files from modified date

    for file in files:
        if file.endswith(".npy"):

            assesed_files += 1
            dataset = np.load(root + "/" + file)
            dataset_len = dataset.shape[0]  # dataset length (approximately 2500) do not depend on this.

            for init_index in range(0, dataset_len,
                                    step_size):  # iterate through dataset while extracting 256 data points

                data_frame = dataset[init_index:init_index + N]

                if data_frame.shape[0] < N:
                    break

                for i_chan in range(8):  # iterate through channel data. (8 channels)

                    channel_data = data_frame[:, i_chan]

                    fft_data = custom_transform.perform_FFT(channel_data, N, pass_high=15, pass_low=50,
                                                            zero_filler=False)  # Transform
                    extracted_data = np.hstack(([i_chan + 1], fft_data, DATA.ACTIONS[dir_name]))  # horizontal stacking

                    if assesed_files < max_train_count:
                        train_2_wr = np.vstack([train_2_wr, extracted_data])  # vertical stacking
                    else:
                        test_2_wr = np.vstack([test_2_wr, extracted_data])
                        if file not in test_set_files:
                            test_set_files.append(file)

        df = pd.DataFrame(data=train_2_wr, columns=df_columns)  # create new dataframe for train data
        path = f"../dataset/transformed_data/fft_data/train/{dir_name}_L{lowcut}_H{highcut}.csv"
        df.to_csv(path, index=False)  # write train csv file.

        df = pd.DataFrame(data=test_2_wr, columns=df_columns)  # create new dataframe for test data
        path = f"../dataset/transformed_data/fft_data/test/{dir_name}_L{lowcut}_H{highcut}.csv"
        df.to_csv(path, index=False)  # write test csv file.
