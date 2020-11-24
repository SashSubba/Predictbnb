#%%

from ast import literal_eval
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

def convert_date_to_year(x):
    try:
        return float(x[:4])
    except:
        return None

#load total listings from csv file onto pd object
data = pd.read_csv('listings.csv')

#initial data information and data separation
#data.info()
columns_list = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,18,19,21,24,26,28,29,30,32,34,39,41,42,43,44,45,46,47,48,49,50,51,52,53,54,56,57,58,59,67,69,70,71,72,73]

X_pd = data.drop(columns=data.columns[columns_list],axis=1)
y_pd = data['price']

encoded_neighbourhood = pd.get_dummies(X_pd.neighbourhood_cleansed, prefix='Neighbourhood')
X_pd = X_pd.drop('neighbourhood_cleansed',axis=1)
X_pd = X_pd.join(encoded_neighbourhood)

X_pd['host_since'] = X_pd['host_since'].apply(convert_date_to_year)

X_pd.info()


#pd object conversion to np for easy data manipulation
X_np = X_pd.to_numpy()
y_np = y_pd.to_numpy()

for i in range(len(X_np[0])):
    print(X_np[0][i])

#%%

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_np, y_np, test_size=0.30, random_state=0)