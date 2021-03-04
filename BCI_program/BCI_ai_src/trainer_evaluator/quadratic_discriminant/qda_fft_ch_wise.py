"""
Quadratic Discriminant Analysis:
    fft representation
    this is for i th channel:
        X = [value10 ... value60]
        y = [label] 
        
@author: Prophet X
"""

#scikit imports
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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

fft_train = '../../dataset/transformed_data/fft_data/train' #fft training data directory
fft_test =  '../../dataset/transformed_data/fft_data/test'  #fft test data directory
model_root = '../../models/qda_models/qda_model1/'          #address for store trained models
          
# seperate random forest algorithms for 8 channels.
qda_channel1 = QuadraticDiscriminantAnalysis()
qda_channel2 = QuadraticDiscriminantAnalysis()
qda_channel3 = QuadraticDiscriminantAnalysis()
qda_channel4 = QuadraticDiscriminantAnalysis()
qda_channel5 = QuadraticDiscriminantAnalysis()
qda_channel6 = QuadraticDiscriminantAnalysis()
qda_channel7 = QuadraticDiscriminantAnalysis()
qda_channel8 = QuadraticDiscriminantAnalysis()

qda_channels = [qda_channel1 , qda_channel2 , qda_channel3 , qda_channel4 , 
                  qda_channel5 , qda_channel6 , qda_channel7 , qda_channel8]

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
                y = np.hstack(  (y, np.array(filtered_df['direction']) ))
                
                csv_count = csv_count+1
    
    if csv_count==0:
        print('No data for train')
        break;
        
    qda_channels[channel].fit(X,y)
                                                    
    # save the models
    model_name = 'channel'+str(channel+1)+'.joblib'
    joblib.dump(qda_channels[channel], model_root+model_name)

#evaluations
total_scores = []    
total_accuracies =[]

    
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
    
    start_time = datetime.datetime.now()           
    predicted = qda_channels[channel].predict(X_test)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    prediction_rate =  float(elapsed_time.microseconds)/len(predicted)
    print('elapsed time for prediction (us)',elapsed_time.microseconds,'\n','prediction rate ',prediction_rate)
                                                
    #cross validation score.
    print('Score for channel ',channel)
    scores = np.average(cross_val_score(qda_channels[channel], X_test, y_test, cv=10))
    report = classification_report(y_test, predicted)
    accuracy = accuracy_score(y_test,predicted)
    total_scores.append(scores)
    total_accuracies.append(accuracy)
    
print(total_accuracies,'\n',total_scores)
