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

#%%
X = prep.X_pd
y = prep.y_pd.astype('string').apply(lambda x: x.str.strip('$')).apply(lambda x: x.str.replace(',', '')).astype('float')

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
fs = SelectKBest(score_func=mutual_info_regression, k=50)
fs.fit(X_scaled_train, y_train)
X_train_fs = fs.transform(X_scaled_train)
X_test_fs = fs.transform(X_test)

for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

#%%
linear_model = sklearn.linear_model.LinearRegression()
print("training accuracy: " + str(linear_model.fit(X_train_fs,y_train).score(X_train_fs,y_train)*100) + "%")

#%% Ensembling
for n_estimator in [1,2,3,4,5,6,7,8,9,10,20,50,100]:
    random_forest_regressor = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimator, random_state=0, max_depth=10)
    random_forest_regressor.fit(X_scaled_train, y_train)
    # predictions = random_forest_regressor.predict(X_test)
    scores = cross_val_score(random_forest_regressor, X_scaled_train, y_train, cv=10 )
    # predictions = cross_val_predict(random_forest_regressor, X_test, y_test)
    accuracy_score = random_forest_regressor.score(X_scaled_test, y_test)
    print("CV score:" , str(scores.mean()*100), "prediction accuracy_score", str(accuracy_score*100))
