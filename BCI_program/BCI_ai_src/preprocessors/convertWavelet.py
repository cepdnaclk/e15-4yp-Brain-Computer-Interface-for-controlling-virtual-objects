# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:12:32 2020

@author: Prophet X
"""
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../utilities')
import custom_transform
import DATA
plt.close('all')

root_raw = '../dataset/lsl_data'
root_transformed = '../dataset/transformed_data'
root_dfft = '../dataset/downloaded_fft_dataset/model_data/data'
root_lsl = '../dataset/lsl_data/left'

fs = 250.0              #sample rate Hz
N = 256                 #number of samples.
col_count = 60
step_size = 200
directory = os.path.join(root_raw)

#create dataframe name
df_columns = ['channel']
for i in range(22):
    frequency_bin = 'cD4_'+str(i+1)  #
    df_columns.append(frequency_bin)
for i in range(38):
    frequency_bin = 'cD3_'+str(i+1)  #
    df_columns.append(frequency_bin)
df_columns.append('direction')

for root,dirs,files in os.walk(root_raw):
    
    if len(files)==0:                           #if directory empty skip that
        continue
    
    dir_name = root.split('\\')[-1]             #get the directory name of the direction
    train_2_wr = np.empty((0,col_count+2))               #empty mat for train data
    test_2_wr = np.empty((0,col_count+2))                #empty mat for test data
    max_train_count = int(len(files)*90/100)    #maximum number of training files.
    assesed_files = 0                           #currently accessed files
    
    for file in files:
        
        if file.endswith(".npy") :
            
            assesed_files+=1  
            dataset = np.load(root+"/"+file)
            dataset_len  = dataset.shape[0]             # dataset length (approximately 2500) do not depend on this.
           
            for init_index in range(0,dataset_len,step_size):  #iterate through dataset while extracting 256 data points
               
                data_frame = dataset[init_index:init_index+N]
                
                if data_frame.shape[0]<N:
                        break
                
                for i_chan in range(8):       #iterate through channel data. (8 channels)
                    
                    channel_data = data_frame[:,i_chan]
                    
                    dwt_data = custom_transform.perform_wavelet (channel_data)
                    extracted_data = np.hstack(([i_chan+1],dwt_data,DATA.ACTIONS[dir_name]))
                    if assesed_files < max_train_count:
                        train_2_wr = np.vstack([train_2_wr,extracted_data])               #vertical stacking
                    else:
                        test_2_wr = np.vstack([test_2_wr,extracted_data])

        df = pd.DataFrame(data=train_2_wr , columns=df_columns )         #create new dataframe for train data
        path = f"../dataset/transformed_data/wavelet_data/train/{dir_name}_S{step_size}_N{N}.csv" 
        df.to_csv(path,index=False)                                      #write train csv file.

        df = pd.DataFrame(data=test_2_wr , columns=df_columns )         #create new dataframe for test data
        path = f"../dataset/transformed_data/wavelet_data/test/{dir_name}_S{step_size}_N{N}.csv" 
        df.to_csv(path,index=False)                                     #write test csv file.
