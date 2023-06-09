# -*- coding: utf-8 -*-
"""Speed_prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ujFFdc-zewKaH5s69TJ7P5Qo46C380lP
"""

from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import geopy.distance
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler,Normalizer
import plotly.graph_objects as go
np.random.seed(1)
tf.random.set_seed(1)
import dvc.api

# Read the DVC-tracked data using DVC
data_path = 'MLOPS-FYP/model_data_preprocess.csv'
with dvc.api.open(data_path) as file:
    # Use the data in your code
    df = file.read()

# Rest of your speed_prediction.py code
# ...

#df = pd.read_csv('model_data_preprocess.csv')

import pandas as pd

# Assuming your DataFrame is named 'df'
columns_to_check = ['distance', 'time_diff', 'segment_speed']

# Check for negative values in the specified columns
has_negative_values = (df[columns_to_check] < 0).any()

# Display columns with negative values
columns_with_negative_values = has_negative_values[has_negative_values].index
print("Columns with negative values:")
print(columns_with_negative_values)

sort_ships = df.groupby(['IMO', 'MMSI', 'VesselName']).size().reset_index(name='counts')

#[[ ]]
test_ships = sort_ships.sample(n=6, random_state=42)
Train_ships = sort_ships[~sort_ships['IMO'].isin(test_ships['IMO'])]


X_train = df[df['VesselName'].isin(Train_ships['VesselName'])]
X_test = df[df['VesselName'].isin(test_ships['VesselName'])]

X_train = X_train.drop(index=X_train[X_train['distance'] == 0.0].index)

# drop rows where column 'B' has a value of 0.0
X_train = X_train.drop(index=X_train[X_train['time_diff'] == 0.0].index)

# drop rows where column 'B' has a value of 0.0
X_train = X_train.drop(index=X_train[X_train['segment_speed'] == 0.0].index)

X_test = X_test.drop(index=X_test[X_test['distance'] == 0.0].index)

# drop rows where column 'B' has a value of 0.0
X_test = X_test.drop(index=X_test[X_test['time_diff'] == 0.0].index)

# drop rows where column 'B' has a value of 0.0
X_test = X_test.drop(index=X_test[X_test['segment_speed'] == 0.0].index)


X_test = X_test.dropna()
X_train = X_train.dropna()
#X_train['Anomaly']=0

X_train['segment_speed'] = abs(X_train['segment_speed'])

X_test['segment_speed'] = abs(X_test['segment_speed'])

# Step 1: Split the labeled dataset into training and validation sets
train_data = X_train[['SourceLat', 'SourceLon', 'DestLat', 'DestLon', 'LAT', 'LON', 'Heading', 'SOG', 'distance', 'time_diff']].values

train_labels = X_train['segment_speed']


data_scaler = Normalizer()
train_data_scaled = data_scaler.fit_transform(train_data)

# Step 3: Model Training
regressor = RandomForestRegressor()
regressor.fit(train_data_scaled, train_labels)

test_data = X_test[['SourceLat', 'SourceLon', 'DestLat', 'DestLon', 'LAT', 'LON', 'Heading', 'SOG', 'distance', 'Month','DayOfWeek','Hour','Minute','Second']].values
test_labels = X_test['segment_speed']
test_data_scaled = data_scaler.transform(test_data)
segment_speed_pred = regressor.predict(test_data_scaled)

import math
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mse = mean_squared_error(test_labels, segment_speed_pred)
mae = mean_absolute_error(test_labels, segment_speed_pred)
rmse = math.sqrt(mse)
r2 = r2_score(test_labels, segment_speed_pred)
print(' Training MSE: ',mse)
print('Training MAE: %.3f' % mae)
print('Training RMSE:', rmse)
print('Training R-Square',r2)
