# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 15:19:16 2021

@author: Prophet X
"""

#scikit imports
from catboost import Pool, CatBoostClassifier,Pool
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

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
from sklearn.model_selection import GridSearchCV

fft_train = '../../dataset/transformed_data/fft_data/train' #fft training data directory
fft_test =  '../../dataset/transformed_data/fft_data/test'  #fft test data directory    
model_root = "../../models/cat_models/"

#-------------------------------------------------------create Training dataset-----------------------------------------------

X = []
y = []
csv_count = 0

for root,dirs,files in os.walk(fft_train):
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

#----------------------------------------------------- creating test data set--------------------------------------------------
X_test = []
y_test = []
csv_count = 0

for root,dirs,files in os.walk(fft_test):
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
    print('No data for test')
    sys.exit()    
#------------------------------------------------------------------------------------------------------------------------------    

cat = CatBoostClassifier(loss_function='MultiClass',
                         logging_level='Silent',
                         random_state=0,
                         task_type = 'GPU',
                         learning_rate = 0.03,
                         eval_metric = 'AUC')

selector = SelectFromModel(estimator=cat,threshold='mean')
scaler = StandardScaler()

    
scaler.fit(X)       #scaler.
X = scaler.transform(X)       
selector.fit(np.array(X),np.array(y))
X = selector.transform(X)


score = np.average(cross_val_score(cat,X,y, cv=10))
print('shape : ',np.array(X).shape)
print('cvs : 10 fold: ',score)

#--------------------------------------------------------------------------------------------------------------------
X_test = scaler.transform(X_test)            
X_test = selector.transform(X_test)
#------------------------------------------------predict for the test set-------------------------------------------


eval_dataset = Pool(X_test,
                    y_test)

cat.fit(X,y,eval_set = eval_dataset,verbose=False)

print(cat.get_best_iteration())


start_time = datetime.datetime.now()           
predicted = cat.predict(X_test)
end_time = datetime.datetime.now()

elapsed_time = end_time - start_time   #elapsed time
prediction_rate =  float(elapsed_time.microseconds)/len(predicted)  #time per one prediction
print('elapsed time for prediction (us)',elapsed_time.microseconds,'\n','time per one instance prediction(uV) ',prediction_rate)                                           

#----------------------------------------------- calculate scores ---------------------------------------------------
cv_scores = cross_val_score(cat, X_test, y_test , cv=10)
std_dev = stdev(cv_scores)                                  #standard deviation
report = classification_report(y_test, predicted)           #classification report
report2 = classification_report(y_test, predicted,output_dict=True)#classification report
accuracy = accuracy_score(y_test, predicted)                #accuracies.
matrix = plot_confusion_matrix(cat, X_test, y_test,         #confusion matrix.
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 display_labels = ['left','right','none'])
plt.title('Confusion matrix for Catboost algorithm')
    
print(report,'\n',accuracy,'\n',std_dev)









