# -*- coding: utf-8 -*-
"""
RandomForest data format:
    fft representation
    this is for i th channel:
        X = [value13 ... value60]
        y = [label]        
@author: Prophet X
"""

#scikit imports
from sklearn.ensemble import RandomForestClassifier
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
model_root = "../../models/rf_models/rf_model1/"

random_forests = []
selectors = []
          
# seperate random forest algorithms for 8 channels.
for channel in range(8):
    # create classification object.
    rfc = RandomForestClassifier(n_estimators=120, random_state=1,
                             n_jobs=-1,
                             max_features='log2',
                             min_samples_leaf=2,
                             min_samples_split=5,
                             max_depth=10)
    random_forests.append(rfc)
    #create selector object.
    selector = SelectFromModel(estimator=random_forests[channel],threshold='mean')
    #selector = RFE(estimator = random_forests[channel], n_features_to_select=18, step=1)
    selectors.append(selector)


# acquire the matrix related to training the algorithm
for channel in range(8):
    print('channel ',channel+1,' Training.')
    X = np.empty((0,37)) 
    y = np.empty((0)) 
    
    csv_count = 0 
    #-------------------------------------------------------create dataset-----------------------------------------------
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
    #feature selection.
    selector = selectors[channel].fit(X,y)
    #print('estimator coefficient : ',selector.estimator_.feature_importances_)
    #print('Threshold             : ',selector.threshold_)
    
    transformed_X = selector.transform(X)
    
    random_forests[channel].fit(transformed_X,y)
    
    score = np.average(cross_val_score(random_forests[channel],transformed_X,y, cv=10))
    print('shape : ',X.shape)
    print('cvs : 10 fold: ',score)
                                                    
    # save the models
    model_name = 'channel'+str(channel+1)+'.joblib'
    joblib.dump(random_forests[channel], model_root+model_name)

#evaluations
total_scores = []
total_accuracies = []    
    
for channel in range(8):
    print('---------------------------------------- channel ',channel+1,' Testing.-----------------------------------------------')
    X_test = np.empty((0,37))
    y_test = np.empty((0)) 
    
    csv_count = 0 
    
    #---------------------------------------------------------- create test set ----------------------------------------
    for root,dirs,files in os.walk(fft_test):
        for file in files:
            if file.endswith(".csv"):
                data_frame = pd.read_csv(root+'/'+file)
                
                filtered_df  = data_frame[data_frame['channel']==(channel+1)]
            
                X_test = np.vstack(  (X_test, np.array(filtered_df.drop(['channel','direction'],axis = 1)  ))   )
                y_test = np.hstack(  (y_test, np.array(filtered_df['direction']) ) ).astype(np.int32)   
                
                csv_count = csv_count + 1
    
    if csv_count == 0 :
        print('No files for test')
        break;
    
    #--------------------------------------------------------------------------------------------------------------------
               
    selector = selectors[channel]
    X_test = selector.transform(X_test)
    
    #------------------------------------------------predict for the test set-------------------------------------------
    start_time = datetime.datetime.now()           
    predicted = random_forests[channel].predict(X_test)
    end_time = datetime.datetime.now()
    
    elapsed_time = end_time - start_time   #elapsed time
    prediction_rate =  float(elapsed_time.microseconds)/len(predicted)  #time per one prediction
    print('elapsed time for prediction (us)',elapsed_time.microseconds,'\n','time per one instance prediction(uV) ',prediction_rate)
                                                
    #cross validation score.
    print('Score for channel ',channel+1)
    
    #----------------------------------------------- calculate scores ---------------------------------------------------
    cv_scores = cross_val_score(random_forests[channel], X_test, y_test, cv=10)
    scores = np.average(cv_scores)                              #cross validation score
    std_dev = stdev(cv_scores)                                  #standard deviation
    report = classification_report(y_test, predicted)           #classification report
    accuracy = accuracy_score(y_test, predicted)                #accuracies.
    
    total_scores.append(scores)
    total_accuracies.append(accuracy)
    print('shape : ',X_test.shape)    
    
print(total_accuracies,'\n',total_scores,'\n',std_dev)
    



