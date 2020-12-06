from ast import literal_eval
import ast
import csv
import numpy as np
import pandas as pd

#Convert date data to only have the year
def convert_date_to_year(x):
    try:
        return float(x[:4])
    except:
        return None

#Convert lists in host_Verifications to their lengths
def convert_list_to_lengths(x):
    try:   
        return len(ast.literal_eval(x))
    except:
        return 0

#load total listings from csv file onto pd object and drop rows with missing values
data = pd.read_csv('listings.csv')

#initial data information and data separation
columns_list = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,18,19,20,21,23,24,26,28,31,32,34,41,42,43,44,45,46,47,48,49,50,51,52,53,54,56,57,58,59,67,68,73]

data = data.drop(columns=data.columns[columns_list],axis=1)
data = data.dropna()

X_pd = data.drop('price',axis=1)
y_pd = data[['price']].astype('string')
y_pd = y_pd.apply(lambda x: x.str.strip('$')).apply(lambda x: x.str.replace(',', ''))



#Preprocess values of host_since to only output the year instead of the whole date
X_pd['host_since'] = X_pd['host_since'].apply(convert_date_to_year)

#Convert host_is_superhost booleans to 1 or 0
X_pd["host_is_superhost"] = X_pd["host_is_superhost"].apply(lambda x : 1 if x == 't' else 0)

#Convert host_identity_verfied booleans to 1 or 0
X_pd["host_identity_verified"] = X_pd["host_identity_verified"].apply(lambda x : 1 if x == 't' else 0)

#Hot encode neighbourhood_cleansed column 
encoded_neighbourhood = pd.get_dummies(X_pd.neighbourhood_cleansed, prefix='Neighbourhood')
X_pd = X_pd.drop('neighbourhood_cleansed',axis=1)
X_pd = X_pd.join(encoded_neighbourhood)

#Preprocess values of bathrooms_text column with their number values
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^20", 20, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^11.5", 11.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^7.5", 7.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^6.5", 6.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^5.5", 5.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^4.5", 4.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^3.5", 3.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^2.5", 2.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^1.5", 1.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^8", 8, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^7", 7, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^6", 6, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^5", 5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^4", 4, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^3", 3, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^2", 2, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^1\s", 1, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^0", 0, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^Private", 0.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^Half", 0.5, regex=True)
X_pd['bathrooms_text'] = X_pd['bathrooms_text'].replace("^half", 0.5, regex=True)

#Preprocess values of amenities to only output the amount of amenities instead of the list of amenities
X_pd["amenities"] = X_pd["amenities"].apply(convert_list_to_lengths)


X_pd = X_pd.astype('float')
y_pd = y_pd.astype('float')