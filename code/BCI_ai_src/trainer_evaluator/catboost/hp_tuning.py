# -*- coding: utf-8 -*-
"""
Catboost algorithm data format:
    fft representation
    this is for i th channel:
        X = [value13 ... value60]
        y = [label]        
@author: Prophet X
"""

# scikit imports
from catboost import Pool, CatBoostClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

# pandas imports
import pandas as pd
import numpy as np
import os, sys
import datetime

# import evaluators
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from statistics import stdev
from sklearn.model_selection import GridSearchCV

fft_train = '../../dataset/transformed_data/fft_data/train'  # fft training data directory
fft_test = '../../dataset/transformed_data/fft_data/test'  # fft test data directory
model_root = "../../models/cat_models/"

# -------------------------------------------------------create Training dataset-----------------------------------------------

X = []
y = []
csv_count = 0

for root, dirs, files in os.walk(fft_train):
    for file in files:
        if file.endswith(".csv"):
            data_frame = pd.read_csv(root + '/' + file)
            for i in range(0, data_frame.shape[0], 8):
                train_x = np.array(data_frame.drop(['channel', 'direction'], axis=1))[i:(i + 8), :].flatten()
                train_y = np.array(data_frame['direction'])[0]
                X.append(train_x)
                y.append(train_y)
        csv_count += 1

if csv_count == 0:
    print('No data for train')
    sys.exit()

# ----------------------------------------------------- creating test data set--------------------------------------------------
X_test = []
y_test = []
csv_count = 0

for root, dirs, files in os.walk(fft_test):
    for file in files:
        if file.endswith(".csv"):
            data_frame = pd.read_csv(root + '/' + file)
            for i in range(0, data_frame.shape[0], 8):
                train_x = np.array(data_frame.drop(['channel', 'direction'], axis=1))[i:(i + 8), :].flatten()
                train_y = np.array(data_frame['direction'])[0]
                X_test.append(train_x)
                y_test.append(train_y)
            csv_count += 1

if csv_count == 0:
    print('No data for test')
    sys.exit()
# ------------------------------------------------------------------------------------------------------------------------------

# number of trees in random forest.
iterations = [100, 120, 150, 200, 250, 300]
learning_rate = [0.03, 0.05, 0.5, 0.8, 0.9, 1.0]
depth = [int(x) for x in range(10, 110, 10)]
random_state = 42

# create random grid
random_grid = {
    'iterations': iterations,
    'learning_rate': learning_rate,
    'depth': depth,
}

cat = CatBoostClassifier(loss_function='MultiClass',
                         logging_level='Silent',
                         random_state=0, task_type='GPU')

selector = SelectFromModel(estimator=cat, threshold='mean')
scaler = StandardScaler()

scaler.fit(X)  # scaler.
X = scaler.transform(X)
selector.fit(np.array(X), np.array(y))
X = selector.transform(X)

cat_gridcv = GridSearchCV(estimator=cat, param_grid=random_grid, cv=5, verbose=2, n_jobs=-1)
cat_gridcv.fit(X, y)
print(cat_gridcv.best_params_)
print(cat_gridcv.best_score_)

score = np.average(cross_val_score(cat, X, y, cv=10))
print('shape : ', np.array(X).shape)
print('cvs : 10 fold: ', score)

# --------------------------------------------------------------------------------------------------------------------
X_test = scaler.transform(X_test)
X_test = selector.transform(X_test)
# ------------------------------------------------predict for the test set-------------------------------------------
start_time = datetime.datetime.now()
predicted = cat.predict(X_test)
end_time = datetime.datetime.now()

elapsed_time = end_time - start_time  # elapsed time
prediction_rate = float(elapsed_time.microseconds) / len(predicted)  # time per one prediction
print('elapsed time for prediction (us)', elapsed_time.microseconds, '\n', 'time per one instance prediction(uV) ',
      prediction_rate)

# ----------------------------------------------- calculate scores ---------------------------------------------------
#std_dev = stdev(cv_scores)  # standard deviation
report = classification_report(y_test, predicted)  # classification report
report2 = classification_report(y_test, predicted, output_dict=True)  # classification report
accuracy = accuracy_score(y_test, predicted)  # accuracies.
matrix = plot_confusion_matrix(cat, X_test, y_test,  # confusion matrix.
                               cmap=plt.cm.Blues,
                               normalize='true',
                               display_labels=['left', 'right', 'none'])
plt.title('Confusion matrix for Catboost algorithm')

print(report, '\n', accuracy, '\n')
