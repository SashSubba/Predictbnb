#%%
import preprocessing as prep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics
import sklearn.svm
import scipy
import scipy.stats
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

#%%
#Preprocessed data retrieval
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
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)

for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

#%%
#Linear Regression Model
linear_model = sklearn.linear_model.LinearRegression()
linear_model.fit(X_train_fs,y_train)
lin_preds = linear_model.predict(X_train_fs)

print('Training Mean Squared Error for Linear Regression: %.2f' % mean_squared_error(y_train, lin_preds))
print('Training R-Squared Score for Linear Regression: %.3f' % linear_model.score(X_train_fs,y_train))

linear_model2 = sklearn.linear_model.LinearRegression()
linear_model2.fit(X_train_fs,y_train)
lin_preds2 = linear_model2.predict(X_test_fs)

print('Testing Mean Squared Error for Linear Regression: %.2f' % mean_squared_error(y_test, lin_preds2))
print('Testing R-Squared Score for Linear Regression: %.3f' % linear_model2.score(X_test_fs,y_test))

plt.scatter(y_test, lin_preds2,  color='black')
#plt.plot(preds, preds, color='blue', linewidth=3)
plt.title("Price predictions for LinearRegression model");
plt.show()

#%%
#Support Vector Regression Model
svr_model = sklearn.svm.LinearSVR(random_state=0)
svr_model.fit(X_train_fs, y_train)
svr_preds = svr_model.predict(X_train_fs)


print('Training Mean Squared Error for Support Vector Regression: %.2f' % mean_squared_error(y_train, svr_preds))
print('Training R-Squared Score for Support Vector Regression: %.3f' % svr_model.score(X_train_fs,y_train))

svr_model2 = sklearn.svm.LinearSVR(random_state=0)
svr_model2.fit(X_train_fs, y_train)
svr_preds2 = svr_model2.predict(X_test_fs)

print('Testing Mean Squared Error for Support Vector Regression: %.2f' % mean_squared_error(y_test, svr_preds2))
print('Testing R-Squared Score for Support Vector Regression: %.3f' % svr_model2.score(X_test_fs,y_test))

#%% 
#Ensembling
rfr = sklearn.ensemble.RandomForestRegressor(random_state=0)
param_distribution = {'bootstrap': [True, False], 'n_estimators': [50,100,200,300,1000], 'max_features': ['auto','sqrt'], 'max_depth':[10,20,30,40,50,100]}
randomized_search_rfr = sklearn.model_selection.RandomizedSearchCV(rfr, param_distribution, n_iter=22, verbose=2, cv=5, random_state=0)
randomized_search_rfr.fit(X_train_fs, y_train)
# print("training accuracy: " + str(rfr.fit(X_train_fs,y_train).score(X_test_fs, y_test)*100) + "%")
# %%
best_esimtator = randomized_search_rfr.best_estimator_
random_search_train_accuracy =  best_esimtator.score(X_test_fs, y_test)*100

print(random_search_train_accuracy)
# 8.834% which took about 22 minutes
# %%
