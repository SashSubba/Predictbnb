#%%

import preprocessing as prep
import matplotlib as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import torch

#%%

X = prep.X_pd
y = prep.y_pd.astype('string').apply(lambda x: x.str.strip('$')).apply(lambda x: x.str.replace(',', '')).astype('float')

linear_model = sklearn.linear_model.LinearRegression()
linear_model.fit(X, y)

# %%
