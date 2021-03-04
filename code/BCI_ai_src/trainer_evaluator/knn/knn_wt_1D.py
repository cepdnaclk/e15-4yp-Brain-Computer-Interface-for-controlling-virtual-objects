# -*- coding: utf-8 -*-

"""
knn algorithm data format:
    fft representation
    this is for i th channel:
        X = [value13 ... value60]
        y = [label]        
@author: Prophet X
"""

#scikit imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,mutual_info_classif

#pandas imports
import pandas as pd
import numpy as np
import joblib
import os,sys
import datetime

#import evaluators
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from statistics import stdev

wt_train = '../../dataset/transformed_data/wavelet_data/train' #fft training data directory
wt_test =  '../../dataset/transformed_data/wavelet_data/test'  #fft test data directory
model_root = '../../models/cat_models/cat_model2/'          #address for store trained models

svm = KNeighborsClassifier(n_neighbors=3)
scaler = StandardScaler()
selector = SelectKBest(score_func = mutual_info_classif, k='all')

X = []
y = []
csv_count = 0
#-------------------------------------------------------create dataset-----------------------------------------------
for root,dirs,files in os.walk(wt_train):
    for file in files:
        if file.endswith(".csv"):
            data_frame = pd.read_csv(root+'/'+file)
            for i in range(0,data_frame.shape[0],8): 
                train_x = np.array(data_frame.drop(['channel','direction'],axis = 1))[i:(i+8),:].flatten()
                train_y = np.array(data_frame['direction'])[0]
                X.append(train_x)
                y.append(train_y)
        csv_count+=1
                
if csv_count==0:
    print('No data for train')
    sys.exit()
           
scaler.fit(X)       #scaler.
transformed_X = scaler.transform(X)
selector.fit(np.array(transformed_X), np.array(y))

transformed_X = selector.transform(transformed_X)     
svm.fit(transformed_X,y)

score = np.average(cross_val_score(svm,transformed_X,y, cv=10))
print('shape : ',np.array(X).shape)
print('cvs : 10 fold: ',score)
  
X_test = []
y_test = []
csv_count = 0

for root,dirs,files in os.walk(wt_test):
    for file in files:
        if file.endswith(".csv"):
            data_frame = pd.read_csv(root+'/'+file)
            for i in range(0,data_frame.shape[0],8): 
                train_x = np.array(data_frame.drop(['channel','direction'],axis = 1))[i:(i+8),:].flatten()
                train_y = np.array(data_frame['direction'])[0]
                X_test.append(train_x)
                y_test.append(train_y)
            csv_count+=1       
                
if csv_count==0:
    print('No data for train')
    sys.exit()

#--------------------------------------------------------------------------------------------------------------------
X_test = scaler.transform(X_test)    
X_test = selector.transform(X_test)  

#------------------------------------------------predict for the test set-------------------------------------------
start_time = datetime.datetime.now()           
predicted = svm.predict(X_test)
end_time = datetime.datetime.now()

elapsed_time = end_time - start_time   #elapsed time
prediction_rate =  float(elapsed_time.microseconds)/len(predicted)  #time per one prediction
print('elapsed time for prediction (us)',elapsed_time.microseconds,'\n','time per one instance prediction(uV) ',prediction_rate)
                                            
#----------------------------------------------- calculate scores ---------------------------------------------------
cv_scores = cross_val_score(svm, X_test, y_test , cv=10)
scores = np.average(cv_scores)                              #cross validation score
std_dev = stdev(cv_scores)                                  #standard deviation
report = classification_report(y_test, predicted)           #classification report
accuracy = accuracy_score(y_test, predicted)                #accuracies.
matrix = plot_confusion_matrix(svm, X_test, y_test,         #confusion matrix.
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for KNN')
    
print(report,'\n',accuracy,'\n',scores,'\n',std_dev)
    






