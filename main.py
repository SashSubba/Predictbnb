#%%
import preprocessing as prep
import matplotlib as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

import sklearn.ensemble
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import scipy
import scipy.stats   
import sklearn.metrics

import sklearn.neural_network
#%%
X = prep.X_pd
y = prep.y_pd

X_np = X.to_numpy()
y_np = y.to_numpy()
    
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_np, y_np, test_size=0.30, random_state=0)

#%%
#Scaling and Normalization of data
scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

#%%
# Run Feature Selection
fs = SelectKBest(score_func=mutual_info_regression, k="all")
fs.fit(X_scaled_train, y_train)
X_train_fs = fs.transform(X_scaled_train)
X_test_fs = fs.transform(X_scaled_test)

for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

#%%
linear_model = sklearn.linear_model.LinearRegression()
print("training accuracy: " + str(linear_model.fit(X_train_fs,y_train).score(X_train_fs,y_train)*100) + "%")

#%% RandomizedSearch with RandomForestRegressor . Note : Training takes about 7.5 mins. Please see comments after print statements for the results

rfr = sklearn.ensemble.RandomForestRegressor( random_state=0)
param_distribution = {'n_estimators': [50,100,200,300], 'max_features': ['auto','sqrt'], 'max_depth':[10,20,30,40,50]}
randomized_search_rfr = sklearn.model_selection.RandomizedSearchCV(rfr, param_distribution, n_iter=22, verbose=2, cv=5, random_state=0)
randomized_search_rfr.fit(X_train_fs, y_train)

best_estimator = randomized_search_rfr.best_estimator_

random_search_train_accuracy =  best_estimator.score(X_train_fs, y_train)*100
random_search_test_accuracy =  best_estimator.score(X_test_fs, y_test)*100


y_pred_1 = best_estimator.predict(X_train_fs)
y_pred_2 = best_estimator.predict(X_test_fs)

training_score = best_estimator.score(X_train_fs, y_train)
testing_score = best_estimator.score(X_test_fs, y_test)

mse_train = sklearn.metrics.mean_squared_error(y_train, y_pred_1)
mse_test = sklearn.metrics.mean_squared_error(y_test, y_pred_2)

print("\nrandomized search best estimator")
print("best_params_: %s" % randomized_search_rfr.best_params_)
print("Training Mean Squared Error: %.3f"  %(mse_train))
print("Training R-Squared Error: %.3f"  %(training_score))
print("Testing Mean Squared Error: %.3f"  %(mse_test))
print("Testing R-Squared Error: %.3f"  %(testing_score))


"""

best estimator
best_params_: {'n_estimators': 300, 'max_features': 'sqrt', 'max_depth': 50}
Training Mean Squared Error: 437.618
Training R-Squared Error: 0.925
Testing Mean Squared Error: 3125.089
Testing R-Squared Error: 0.481

"""
# %% RandomizedSearch with Multi-layer Perceptron regressor. Note : Training takes about 4 mins. Please see comments after print statements for the results

nn = sklearn.neural_network.MLPRegressor( random_state=0, momentum=0.9)
param_neural_net = {'solver': ['sgd','adam'], 'hidden_layer_sizes':[(),(100,),(100,57),(57,25),(100,57,25)], 'activation':['tanh','sgd'], 'batch_size':[100,200,300], 'max_iter':[10,50,100,200,500],'learning_rate_init':[0.001, 0.01, 0.1]}
randomized_search_nn = sklearn.model_selection.RandomizedSearchCV(nn, param_neural_net, n_iter=40, verbose=2, cv=5, random_state=0)
randomized_search_nn.fit(X_train_fs, y_train)

best_estimator = randomized_search_nn.best_estimator_

y_pred_1 = best_estimator.predict(X_train_fs)
y_pred_2 = best_estimator.predict(X_test_fs)

training_score = best_estimator.score(X_train_fs, y_train)
testing_score = best_estimator.score(X_test_fs, y_test)

mse_train = sklearn.metrics.mean_squared_error(y_train, y_pred_1)
mse_test = sklearn.metrics.mean_squared_error(y_test, y_pred_2)

print("\n\n neural net search best estimator")
print("best_params_: %s" % randomized_search_nn.best_params_)
print("Training Mean Squared Error: %.3f"  %(mse_train))
print("Training R-Squared Error: %.3f"  %(training_score))
print("Testing Mean Squared Error: %.3f"  %(mse_test))
print("Testing R-Squared Error: %.3f"  %(testing_score))

"""
best estimator
best_params_: {'solver': 'sgd', 'max_iter': 10, 'learning_rate_init': 0.001, 'hidden_layer_sizes': (100, 57), 'batch_size': 100, 'activation': 'tanh'}
Training Mean Squared Error: 3414.409
Training R-Squared Error: 0.418
Testing Mean Squared Error: 3673.520
Testing R-Squared Error: 0.390
"""
