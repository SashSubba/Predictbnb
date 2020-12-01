#%%
import preprocessing as prep
import matplotlib as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import torch

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
#%%

X = prep.X_pd
y = prep.y_pd.astype('string').apply(lambda x: x.str.strip('$')).apply(lambda x: x.str.replace(',', '')).astype('float')

X_np = X.to_numpy()
y_np = y.to_numpy()
    
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_np, y_np, test_size=0.30, random_state=0)

#%% Feature Selection
fs = SelectKBest(score_func=mutual_info_regression, k='all')
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

#%%
linear_model = sklearn.linear_model.LinearRegression()
linear_model.fit(X, y)

# %%
