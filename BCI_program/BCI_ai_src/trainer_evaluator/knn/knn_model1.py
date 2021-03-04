"""
KNN model Analysis:
    fft representation
    this is for i th channel:
        X = [value10 ... value60]
        y = [label] 
        
@author: Prophet X
"""

#scikit imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

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
from statistics import stdev

fft_train = '../../dataset/transformed_data/fft_data/train' #fft training data directory
fft_test =  '../../dataset/transformed_data/fft_data/test'  #fft test data directory
model_root = '../../models/knn_models/knn_model1/'          #address for store trained models

knn_channels = []
selectors = []

for channel in range(8):
    print('channel ',channel+1,' Training.')
    X = np.empty((0,37)) 
    y = np.empty((0)) 
    
    csv_count = 0 
    
    for root,dirs,files in os.walk(fft_train):
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
        
    #--------------------------------------------------------------------------------------------------------------------
        
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_channels.append(knn)
    knn_channels[channel].fit(X,y)
    
    selector = SelectFromModel(estimator = knn_channels[channel],prefit = True,threshold='mean')
    selectors.append(selector)    
        
    transformed_X = selector.transform(X)
    
    knn_channels[channel].fit(transformed_X,y)
    
    score = np.average(cross_val_score(knn_channels[channel],transformed_X,y, cv=10))
    print('shape : ',X.shape)
    print('cvs : 10 fold: ',score)
    
    

#evaluations
total_scores = []
total_accuracies = []
    
for channel in range(8):
    print('channel ',channel+1,' Testing.')
    X_test = np.empty((0,37))
    y_test = np.empty((0)) 
    
    csv_count = 0 
    
    for root,dirs,files in os.walk(fft_test):
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
    
    #--------------------------------------------------------------------------------------------------------------------
    selector = selectors[channel]
    X_test = selector.transform(X_test)
    
    #------------------------------------------------predict for the test set-------------------------------------------
    start_time = datetime.datetime.now()           
    predicted = knn_channels[channel].predict(X_test)
    end_time = datetime.datetime.now()
    
    elapsed_time = end_time - start_time   #elapsed time
    prediction_rate =  float(elapsed_time.microseconds)/len(predicted)  #time per one prediction
    print('elapsed time for prediction (us)',elapsed_time.microseconds,'\n','time per one instance prediction(uV) ',prediction_rate)
                                                
    #cross validation score.
    print('Score for channel ',channel+1)
    
    #----------------------------------------------- calculate scores ---------------------------------------------------
    cv_scores = cross_val_score(knn_channels[channel], X_test, y_test, cv=10)
    scores = np.average(cv_scores)                              #cross validation score
    std_dev = stdev(cv_scores)                                  #standard deviation
    report = classification_report(y_test, predicted)           #classification report
    accuracy = accuracy_score(y_test, predicted)                #accuracies.
    
    total_scores.append(scores)
    total_accuracies.append(accuracy)
    print('shape : ',X_test.shape)   
    
print(total_accuracies,'\n',total_scores)
 