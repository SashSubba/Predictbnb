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

def convert_date_to_year(x):
    try:
        return float(x[:4])
    except:
        return None

#load total listings from csv file onto pd object and drop rows with missing values
data = pd.read_csv('listings.csv')

#initial data information and data separation
columns_list = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,18,19,21,24,26,28,29,30,32,34,41,42,43,44,45,46,47,48,49,50,51,52,53,54,56,57,58,59,67,69,70,71,72,73]
data = data.drop(columns=data.columns[columns_list],axis=1)
data = data.dropna()

X_pd = data.drop('price',axis=1)
y_pd = data[['price']]

encoded_neighbourhood = pd.get_dummies(X_pd.neighbourhood_cleansed, prefix='Neighbourhood')
X_pd = X_pd.drop('neighbourhood_cleansed',axis=1)
X_pd = X_pd.join(encoded_neighbourhood)

X_pd['host_since'] = X_pd['host_since'].apply(convert_date_to_year)

#preprocess values of bathrooms_text column with their number values
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace( np.nan, 0)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace( "0", 0, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^16", 16, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^1\s", 1, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("2", 2, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("2.5", 2.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("3", 2.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("3.5", 3.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("4", 4, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("5", 5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("5.5", 5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("6", 6, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("6.5", 6.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("8", 8, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^Half", 0.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("half", 0.5, regex=True)

#one hot enocde the property_type
encoded_property_types = pd.get_dummies(X_pd["property_type"])

X_pd = X_pd.drop("property_type",axis=1)
X_pd = X_pd.join(encoded_property_types)

#Convert lists in host_Verifications to their lengths
import ast

def convert_list_to_lengths(x):
    try:   
        return len(ast.literal_eval(x))
    except:
        return 0

X_pd["host_verifications"] = X_pd["host_verifications"].apply(convert_list_to_lengths)
X_pd["amenities"] = X_pd["amenities"].apply(convert_list_to_lengths)

#convert host_is_superhost booleans to 1 or 0
X_pd["host_is_superhost"] = X_pd["host_is_superhost"].apply(lambda x : 1 if x == 't' else 0)

#hot encode host_neighbourhood
encoded_host_neighbourhood = pd.get_dummies(X_pd["host_neighbourhood"])

X_pd = X_pd.drop("host_neighbourhood",axis=1)
X_pd = X_pd.join(encoded_host_neighbourhood)

#convert host_identity_verfied booleans
X_pd["host_identity_verified"] = X_pd["host_identity_verified"].apply(lambda x : 1 if x == 't' else 0)

#convert instant_bookable_booleans
X_pd["instant_bookable"] = X_pd["instant_bookable"].apply(lambda x : 1 if x == 't' else 0)

X_pd.info(10)
