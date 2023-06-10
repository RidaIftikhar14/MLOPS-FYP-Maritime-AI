import os
import numpy as np
import pandas as pd
from flask import Flask,redirect,url_for,render_template,request,jsonify,Response
import pickle
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import json
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential
import math
import joblib
import geopy.distance
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


np.random.seed(1)
tf.random.set_seed(1)
app = Flask(__name__)

print('Reading simulation csv')
# df = pd.read_csv('data/complete_journeys.csv')
df = pd.read_csv('data/all_ships.csv')
# df = pd.read_csv('data/model_data_preprocess.csv')

print('Reading timestamp information')
# Intervals = pd.read_csv('data/interval_list.csv')
Intervals = pd.read_csv('data/all_ships_interval_list.csv')

# ANOMALY 1 DATASET///////////////////////////////////////////////
# df_anomalous = pd.read_csv('data/test_route_aliza.csv')
# list_for_models_anomaly_1 = []
list_for_models = []

# PORTS /////////////////////////////////////////////////////////
print('Reading Port information')
ports_df = pd.read_csv('data/ports.csv')
ports_df = ports_df.drop('Unnamed: 0',axis=1)

# SORTING ///////////////////////////////////////////////////////
print('Sorting by time')
Sorted_by_time = df
Sorted_by_time = Sorted_by_time.sort_values('BaseDateTime').reset_index().drop('index',axis=1)

# Sorted_by_time_anomalous_1 = df_anomalous
# Sorted_by_time_anomalous_1 = Sorted_by_time_anomalous_1.sort_values('BaseDateTime').reset_index().drop('index',axis=1)
# SORTED/////////////////////////////////////////////////////////
#This data will be sent at the start of website for ship information
print("Extracting Static data")
Static_data = df
Static_data = Static_data.drop(['BaseDateTime','LAT','LON','SOG','COG','Cargo','TransceiverClass','SourceLat','SourceLon','DestLat','DestLon'],axis=1)
Static_data = Static_data.groupby('MMSI').first()
Static_data = Static_data.reset_index()
Static_data["MMSI"] = Static_data["MMSI"].astype(str)
# print('Extracting timestamps')
# Timer = Sorted_by_time['BaseDateTime'].unique()
Timestamps = 0


# MODELS ////////////////////////////////////////////////////////
anomaly_1_model = tf.keras.models.load_model('models/route_model.h5')


anomaly_2_model = ''
filename = 'models/isolation_forest_model.pkl'
# Load the saved model from the file
with open(filename, 'rb') as file:
    anomaly_2_model = pickle.load(file)


scaler=Normalizer()
bounds = pd.read_csv("models/speed_bounds_update.csv")
# # Saved model file 
# with open('model_rf_v3_norm.pkl', 'rb') as f:
#     rf_regressor = pickle.load(f)

def anomaly_1(df_temp):
    cols_keep=['LAT','LON','Heading','SourceLat','SourceLon','DestLat','DestLon']
    df_temp = df_temp.drop([col for col in df_temp.columns if col not in cols_keep], axis=1)
    df_temp=df_temp.dropna(axis=0)
    if df_temp.shape[0] > 0:
        scaler = StandardScaler()
        data = scaler.fit_transform(df_temp)
        y_pred=1
        anomaly=0
        y_pred=anomaly_1_model.predict(data)
        mse=np.mean(np.square(y_pred-data),axis=1)
        threshold=1.1
        binary_predictions = np.where(mse <= threshold, 1, 0)
        unique_values, value_counts = np.unique(binary_predictions, return_counts=True)
        most_common_value = unique_values[np.argmax(value_counts)]
        if most_common_value == 0:
            return 1
        else:
            return -1
    return 1

def Cargo_Anomaly(data):
  selected_features = ['SourceLat', 'SourceLon', 'DestLat', 'DestLon', 'LAT','LON','distance','segment_speed','VesselType','Cargo']
  data = data[selected_features]
  # Predict anomaly scores for the new data
  anomaly_scores = anomaly_2_model.decision_function(data)
  predictions = anomaly_2_model.predict(data)
  # Step 6: Analyze Results
  anomaly_df = pd.DataFrame({'AnomalyScore': anomaly_scores, 'IsAnomaly': predictions}, index=data.index)
  anomaly_df['IsAnomaly'] = anomaly_df['IsAnomaly']  # Convert -1/1 labels to boolean values
  return anomaly_df['IsAnomaly'].iloc[-1]


def anomaly_2_distance(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6.378e+6  # Radius of earth in kilometers
    return c * r

def anomaly_2_speed(vessel_points1):
    vessel_points1['BaseDateTime'] = pd.to_datetime(vessel_points1['BaseDateTime'])
    vessel_points1 = vessel_points1.sort_values(['BaseDateTime'])
    vessel_points1 = vessel_points1.drop_duplicates(subset=["LAT", "LON"], keep='first')
    vessel_points1 = vessel_points1.dropna()

    vessel_points1['time_diff'] = (vessel_points1['BaseDateTime'].diff().dt.total_seconds())

    for i in range(1, len(vessel_points1)):
        lat1, lon1 = vessel_points1.iloc[i - 1]['LAT'], vessel_points1.iloc[i-1]['LON']
        lat2, lon2 = vessel_points1.iloc[i]['LAT'], vessel_points1.iloc[i]['LON']
        dist = anomaly_2_distance(lat1, lon1, lat2, lon2)
        vessel_points1.at[vessel_points1.index[i], 'distance'] = dist

        vessel_points1['segment_speed'] = (vessel_points1['distance']) / vessel_points1['time_diff']

    # vessel_points1 = vessel_points1.drop(index=vessel_points1[vessel_points1['distance'] == 0.0].index)
    vessel_points1 = vessel_points1.drop(index=vessel_points1[vessel_points1['time_diff'] == 0.0].index)
    # vessel_points1 = vessel_points1.drop(index=vessel_points1[vessel_points1['segment_speed'] == 0.0].index)
    vessel_points1 = vessel_points1.dropna()

    return vessel_points1

def anomaly_2(X_test):
    vessel_data = anomaly_2_speed(X_test)
    X_Test_first = vessel_data[['SourceLat', 'SourceLon', 'DestLat', 'DestLon', 'LAT', 'LON', 'time_diff', 'distance']].values
    X_Test_cargo = vessel_data
    return Cargo_Anomaly(X_Test_cargo)



def anomaly_3(X_test,bounds):
    # # Calculate distance, time difference as model features
    # vessel_data = calculate_speed(X_test)
    # X_Test_first = vessel_data[['SourceLat', 'SourceLon', 'DestLat', 'DestLon','LAT', 'LON', 'Heading','SOG','time_diff', 'distance']].values
    
    # # Save the speed ranges for the Source and Destination of input data
    # filtered_data = bounds[
    #     (bounds['SourceLatitude'] ==   X_test['SourceLat'].iloc[0]) &
    #     (bounds['SourceLongitude'] == X_test['SourceLon'].iloc[0]) &
    #     (bounds['DestLatitude'] == X_test['DestLat'].iloc[0]) &
    #     (bounds['DestLongitude'] == X_test['DestLon'].iloc[0])
    #     ]

    #     # Check if any matching rows exist
    # if not filtered_data.empty:
    #     # Extract upper bound and lower bound
    #     upper_bound = filtered_data['upper_bound'].values[0]
    #     lower_bound = filtered_data['lower_bound'].values[0]

    # #predict speed of vessel it should be moving at
    # X_scaled = scaler.transform(X_Test_first)
    # segment_speed_pred = rf_regressor.predict(X_scaled)
    # data_length = len(X_test) - 1
    # #save anomalous indexes from the original data frame
    # anomalies_indices = np.where((segment_speed_pred > upper_bound) | (segment_speed_pred < lower_bound))[0]
    # #return -1 for anomaly and 0 for no anomaly
    # if anomalies_indices.size > 0:
    #     last_anomaly_index = anomalies_indices[-1]
    #     if last_anomaly_index == data_length:
    #         return -1
    #     else:
    #         return 0
    # else:
    #     return 0
    return 1





#////////////////////////////////////////////////////////////////////////// WEBPAGES //////////////////////////////////////////////////////////////////////////
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),'images/favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/Static_Data')
def Static_Data():
    return Static_data.to_json()

@app.route('/Port_info')
def Port_info():
    return ports_df.to_json()

@app.route('/Current_Time_Information')
def Current_Time_Information():
    global Timestamps
    global list_for_models
    
    temp_df = Sorted_by_time.iloc[Intervals['StartIndex'][int(Timestamps)]:Intervals['EndIndex'][int(Timestamps)]].reset_index().drop('index',axis=1)
    for j in range(0,len(temp_df['MMSI'])):
        index = -1
        temp_val = temp_df['MMSI'][j]
        for k in range(0,len(list_for_models)):
            if list_for_models[k][0] == temp_val:
                index = k
                break
        if index == -1:
            row = temp_df.loc[j]
            new_df = pd.DataFrame(row).transpose()
            temp_list = [temp_val,new_df]
            list_for_models.append(temp_list)
        else:
            s = pd.Series(temp_df.loc[j])
            list_for_models[index][1].loc[len(list_for_models[index][1])] = s.values
            list_for_models[index][1] = list_for_models[index][1].sort_index()
    
    # temp_df_anomalous_1 = Sorted_by_time_anomalous_1.iloc[Intervals['StartIndex'][int(Timestamps)]:Intervals['EndIndex'][int(Timestamps)]].reset_index().drop('index',axis=1)
    # for j in range(0,len(temp_df_anomalous_1['MMSI'])):
    #     index = -1
    #     temp_val = temp_df_anomalous_1['MMSI'][j]
    #     for k in range(0,len(list_for_models_anomaly_1)):
    #         if list_for_models_anomaly_1[k][0] == temp_val:
    #             index = k
    #             break
    #     if index == -1:
    #         row = temp_df_anomalous_1.loc[j]
    #         new_df = pd.DataFrame(row).transpose()
    #         temp_list = [temp_val,new_df]
    #         list_for_models_anomaly_1.append(temp_list)
    #     else:
    #         s = pd.Series(temp_df_anomalous_1.loc[j])
    #         list_for_models_anomaly_1[index][1].loc[len(list_for_models_anomaly_1[index][1])] = s.values
    #         list_for_models_anomaly_1[index][1] = list_for_models_anomaly_1[index][1].sort_index()
    
    temp = Sorted_by_time.iloc[Intervals['StartIndex'][int(Timestamps)]:Intervals['EndIndex'][int(Timestamps)]].reset_index().drop('index',axis=1).to_json()
    Timestamps = Timestamps + 1
    return temp

@app.route('/ai_models')
def ai_models():
    anomalies = []
    for i in range(0,pd.DataFrame(list_for_models).shape[0]):
        predictions = []
        predictions.append(anomaly_1(list_for_models[i][1]))
        if list_for_models[i][1].shape[0] > 2:
            predictions.append(anomaly_2(list_for_models[i][1]))
            predictions.append(anomaly_3(list_for_models[i][1], bounds))
        curr_MMSI = list_for_models[i][0]
        time_stamp = list_for_models[i][1].reset_index().drop("index",axis=1)["BaseDateTime"][0]
        anomalies.append([curr_MMSI, str(time_stamp), predictions])
    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df.to_json()
    # return pd.DataFrame(list_for_models).to_json()

@app.route("/search", methods=["POST"])
def search():
    query = request.data.decode('utf-8')
    index = Static_data.loc[Static_data["MMSI"] == query].index[0]
    row = Static_data.loc[Static_data['MMSI'] == query]
    imo_number = row['IMO'].values[0].astype(str)
    main_folder = imo_number[0:4]
    file_one = imo_number[0:6] + "_1"
    file_two = imo_number[0:6] + "_2"
    file_address_one = 'static/images/ship_images/' + main_folder + '/' + file_one
    file_address_two = 'static/images/ship_images/' + main_folder + '/' + file_two
    addresses = {"address_one" : file_address_one , "address_two" : file_address_two}
    temp_series = pd.Series(addresses)
    temp_data = pd.concat([Static_data.loc[index],temp_series])
    return temp_data.to_json()

#////////////////////////////////////////////////////////////////////////// MAIN //////////////////////////////////////////////////////////////////////////
if __name__=='__main__':
    app.run(debug=True,port=8000)
