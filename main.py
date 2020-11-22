#%%

import csv
import gzip
import json
import matplotlib as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import torch

#%%

#load total listings from csv file onto pd object
data = pd.read_csv('listings.csv');

#initial data information and data separation
data.info();
X_pd = data.drop('price', axis=1);
y_pd = data['price'];

#pd object conversion to np for easy data manipulation
X_np = X_pd.to_numpy();
y_np = y_pd.to_numpy();

#%%

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_np, y_np, test_size=0.30, random_state=0);