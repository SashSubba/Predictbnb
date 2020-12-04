#%%

import preprocessing as prep
import matplotlib as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import torch

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

scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

# %%

#%% Ensembling

for n_estimator in [1,2,3,4,5,6,7,8,9,10,20,50,100]:
    random_forest_regressor = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimator, random_state=0, max_depth=10)
    random_forest_regressor.fit(X_scaled_train, y_train)
    # predictions = random_forest_regressor.predict(X_test)
    scores = cross_val_score(random_forest_regressor, X_scaled_train, y_train, cv=10 )
    # predictions = cross_val_predict(random_forest_regressor, X_test, y_test)
    accuracy_score = random_forest_regressor.score(X_scaled_test, y_test)
    print("CV score:" , str(scores.mean()*100), "prediction accuracy_score", str(accuracy_score*100))
        
# %%
