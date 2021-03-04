#scikit imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz

#pandas imports
import pandas as pd
import numpy as np
import joblib
import os
import datetime

#import evaluators
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,accuracy_score
import seaborn as sns

wt_train = '../../dataset/transformed_data/wavelet_data/train' #fft training data directory
wt_test =  '../../dataset/transformed_data/wavelet_data/test'  #fft test data directory
model_root = '../../models/knn_models/knn_model2/'          #address for store trained models
          
# seperate random forest algorithms for 8 channels.
knn_channel1 = KNeighborsClassifier(n_neighbors=5)
knn_channel2 = KNeighborsClassifier(n_neighbors=5)
knn_channel3 = KNeighborsClassifier(n_neighbors=5)
knn_channel4 = KNeighborsClassifier(n_neighbors=5)
knn_channel5 = KNeighborsClassifier(n_neighbors=5)
knn_channel6 = KNeighborsClassifier(n_neighbors=5)
knn_channel7 = KNeighborsClassifier(n_neighbors=5)
knn_channel8 = KNeighborsClassifier(n_neighbors=5)

knn_channels = [knn_channel1 , knn_channel2 , knn_channel3 , knn_channel4 , 
                  knn_channel5 , knn_channel6 , knn_channel7 , knn_channel8]

for channel in range(8):
    print('channel ',channel+1,' Training.')
    X = np.empty((0,60)) 
    y = np.empty((0)) 
    
    csv_count = 0 
    
    for root,dirs,files in os.walk(wt_train):
        for file in files:
            if file.endswith(".csv"):
                data_frame = pd.read_csv(root+'/'+file)
                
                filtered_df  = data_frame[data_frame['channel']==(channel+1)]
            
                X = np.vstack(  (X, np.array(filtered_df.drop(['channel','direction'],axis = 1)  ))   )
                y = np.hstack(  (y, np.array(filtered_df['direction']) )).astype(np.int32)
                
                csv_count = csv_count+1
    
    if csv_count==0:
        print('No data for train')
        break;
        
    knn_channels[channel].fit(X,y)
                                                    
    # save the models
    model_name = 'channel'+str(channel+1)+'.joblib'
    joblib.dump(knn_channels[channel], model_root+model_name)

#evaluations
total_scores = []  
total_accuracies = []
  
for channel in range(8):
    print('channel ',channel+1,' Testing.')
    X_test = np.empty((0,60))
    y_test = np.empty((0)) 
    
    csv_count = 0 
    
    for root,dirs,files in os.walk(wt_test):
        for file in files:
            if file.endswith(".csv"):
                data_frame = pd.read_csv(root+'/'+file)
                
                filtered_df  = data_frame[data_frame['channel']==(channel+1)]
            
                X_test = np.vstack(  (X_test, np.array(filtered_df.drop(['channel','direction'],axis = 1)  ))   )
                y_test = np.hstack(  (y_test, np.array(filtered_df['direction']) ) )  
                
                csv_count = csv_count + 1
    
    if csv_count == 0 :
        print('No files for test')
        break;
    
    start_time = datetime.datetime.now()           
    predicted = knn_channels[channel].predict(X_test)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    prediction_rate =  float(elapsed_time.microseconds)/len(predicted)
    print('elapsed time for prediction (us)',elapsed_time.microseconds,'\n','prediction rate ',prediction_rate)
                                                
    #cross validation score.
    print('Score for channel ',channel)
    scores = np.average(cross_val_score(knn_channels[channel], X_test, y_test, cv=10))
    report = classification_report(y_test, predicted)
    accuracy = accuracy_score(y_test, predicted)
    total_scores.append(scores)
    total_accuracies.append(accuracy)
    
print(total_accuracies,'\n',total_scores)
 



