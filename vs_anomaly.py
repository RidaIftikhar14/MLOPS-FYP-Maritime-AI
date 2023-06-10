import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Define your LSTM model

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(None, 1),return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])

# Compile the model

model.compile(optimizer='adam', loss='mse')
tr=pd.read_csv('route_github.csv')
vessel=['T.ADALYN','DALI','NAVE GALACTIC','STARDUST','ISLENO','ARRIBA','BEVERLY M I','CHICAGO EXPRESS','ELKA HERCULES','FROSTI','GLOVIS CAPTAIN','GOLDEN ALASKA','HYUNDAI PRIDE','KEN HOU','RUBY M','KODIAK ENTERPRISE','VIDA','WESTWOOD RAINIER']
tr=tr[tr['VesselName'].isin(vessel)]
#tr=tr.drop(['MMSI','IMO','SOG','COG','CallSign','VesselType','Status','Length','Width','Draft','Cargo','TransceiverClass'],axis=1)
tr['BaseDateTime'] = tr['BaseDateTime'].str.replace('T', ' ')
tr=tr.dropna(axis=0)

tr= tr.drop(['VesselName','IMO','SOG','COG','CallSign','VesselType','Status','Length','Width','Draft','Cargo','TransceiverClass','BaseDateTime'],axis=1)
print(tr.columns)
tr=tr.drop(['Unnamed: 0'])

scaler = StandardScaler()

ft=pd.DataFrame()

scaler = StandardScaler()
dg=pd.read_csv('route_github.csv')
vv=dg[dg['VesselName']=='ISLENO']
noise_lat = np.random.normal(0, 5, len(vv)-2)
noise_lon = np.random.normal(0, 5, len(vv)-2)
ft['MMSI']=vv['MMSI']

ft['LAT']=vv['LAT'] + 0
ft['LON']=vv['LON'] + 0
#ft['LAT'][0]=vv['LAT'][0]
ft['Heading']=vv['Heading']
#ft['LON'][0]=vv['LON'][0]
#ft.at[0, 'LAT'] = vv['LAT'][0]
#ft.at[0, 'LON'] = vv['LON'][0]
ft['SourceLat']=vv['SourceLat']
ft['SourceLon']=vv['SourceLon']

ft['DestLat']=vv['DestLat']
ft['DestLon']=vv['DestLon']

#ft.at[0, 'LAT']=vv['LAT'][0]
#ft.at[0, 'LON']=vv['LON'][0]
ft.iloc[1:len(ft)-1, ft.columns.get_loc('LAT')] +=noise_lat

ft.iloc[1:len(ft)-1, ft.columns.get_loc('LON')] +=noise_lon
print(ft.columns)
ft=ft.dropna()

dg=pd.read_csv('route_github.csv')
ft1=pd.DataFrame()
vv=dg[dg['VesselName']=='STARDUST']
noise_lat = np.random.normal(0, 5, len(vv)-2)
noise_lon = np.random.normal(0, 5, len(vv)-2)
ft1['MMSI']=vv['MMSI']

ft1['LAT']=vv['LAT'] 
ft1['LON']=vv['LON'] 
#ft1.at[0,'LAT']=vv['LAT']
ft1['Heading']=vv['Heading']
#ft1.at[0,'LON']=vv['LON']
ft1['SourceLat']=vv['SourceLat']
ft1['SourceLon']=vv['SourceLon']

ft1['DestLat']=vv['DestLat']
ft1['DestLon']=vv['DestLon']

ft1.iloc[1:len(ft1)-1, ft1.columns.get_loc('LAT')] +=noise_lat

ft1.iloc[1:len(ft1)-1, ft1.columns.get_loc('LON')] +=noise_lon
ft1=ft1.dropna()
e=np.full(len(tr),0)
h=np.full(len(ft),1)
hl=np.full(len(ft1),1)
e=np.concatenate((e,h,hl))
tr=pd.concat([tr,ft,ft1])
tr['is_anomaly']=e
normalized_data = scaler.fit_transform(tr)
# Generate training data (replace with your own training data)
train_data = np.random.rand(100, 10, 1)
train_labels = np.random.rand(100, 1)
e=np.full(len(normalized_data),0)
# Train the model
model.fit(normalized_data,e, epochs=12)
model.save('modelroute.h5')
# Generate test data (replace with your own test data)

"""
ft=pd.DataFrame()

scaler = StandardScaler()

dg=pd.read_csv('D:/CompleteJourney_update.csv')
ft1=pd.DataFrame()
vv=dg[dg['VesselName']=='STARDUST']
noise_lat = np.random.normal(0, 0.1, len(vv)-2)
noise_lon = np.random.normal(0, 0.1, len(vv)-2)
ft1['MMSI']=vv['MMSI']

ft1['LAT']=vv['LAT'] 
ft1['LON']=vv['LON'] 
#ft1.at[0,'LAT']=vv['LAT']
ft1['Heading']=vv['Heading']
#ft1.at[0,'LON']=vv['LON']
ft1['SourceLat']=vv['SourceLat']
ft1['SourceLon']=vv['SourceLon']

ft1['DestLat']=vv['DestLat']
ft1['DestLon']=vv['DestLon']

ft1.iloc[1:len(ft1)-1, ft1.columns.get_loc('LAT')] +=noise_lat

ft1.iloc[1:len(ft1)-1, ft1.columns.get_loc('LON')] +=noise_lon
ft1=ft1.dropna()

dg=pd.read_csv('D:/CompleteJourney_update.csv')
vv=dg[dg['VesselName']=='ISLENO']
noise_lat = np.random.normal(0, 8, len(vv)-2)
noise_lon = np.random.normal(0, 8, len(vv)-2)
ft['MMSI']=vv['MMSI']

ft['LAT']=vv['LAT'] + 0
ft['LON']=vv['LON'] + 0
#ft['LAT'][0]=vv['LAT'][0]
ft['Heading']=vv['Heading']
#ft['LON'][0]=vv['LON'][0]
#ft.at[0, 'LAT'] = vv['LAT'][0]
#ft.at[0, 'LON'] = vv['LON'][0]
ft['SourceLat']=vv['SourceLat']
ft['SourceLon']=vv['SourceLon']

ft['DestLat']=vv['DestLat']
ft['DestLon']=vv['DestLon']

#ft.at[0, 'LAT']=vv['LAT'][0]
#ft.at[0, 'LON']=vv['LON'][0]
ft.iloc[1:len(ft)-1, ft.columns.get_loc('LAT')] +=noise_lat

ft.iloc[1:len(ft)-1, ft.columns.get_loc('LON')] +=noise_lon
print(ft.columns)
ft=ft.dropna()
h=np.full(len(ft),1)
hl=np.full(len(ft1),0)
#h1=np.concatenate((hl,h))
#tr=pd.concat([ft1,ft])
km=ft1

tr=pd.read_csv('D:/CompleteJourney_update.csv')
dg=tr
vv=dg[dg['VesselName']=='ISLENO']
noise_lat = np.random.normal(0, 8, len(vv))
noise_lon = np.random.normal(0, 8, len(vv))
ft['MMSI']=vv['MMSI']
ft['LAT']=vv['LAT'] + noise_lat
ft['LON']=vv['LON'] + noise_lon
ft['Heading']=vv['Heading']

ft['SourceLat']=vv['SourceLat']
ft['SourceLon']=vv['SourceLon']

ft['DestLat']=vv['DestLat']
ft['DestLon']=vv['DestLon']


tr=pd.read_csv('D:/CompleteJourney_update.csv')
dg=tr
vv=dg[dg['VesselName']=='STARDUST']
noise_lat = np.random.normal(0, 0.1, len(vv))
noise_lon = np.random.normal(0, 8, len(vv))
ft['MMSI']=vv['MMSI']
ft['LAT']=vv['LAT'] + noise_lat
ft['LON']=vv['LON'] + noise_lon
ft['Heading']=vv['Heading']

ft['SourceLat']=vv['SourceLat']
ft['SourceLon']=vv['SourceLon']

ft['DestLat']=vv['DestLat']
ft['DestLon']=vv['DestLon']

print(ft.columns)


#ft=ft.dropna()
tr = scaler.fit_transform(ft1)
#test_targets=np.full(len(ft),)
# Predict on the test data

model = tf.keras.models.load_model('D:/modelnew1.h5')

df=pd.read_csv('D:/test_route.csv')
cols_keep=['LAT','LON','Heading','SourceLat','SourceLon','DestLat','DestLon']
   
df = df.drop([col for col in df.columns if col not in cols_keep], axis=1)
df=df.dropna(axis=0)
scaler = StandardScaler()
data = scaler.fit_transform(df)


#km.to_csv('D:/test_route4.csv', index=False)
predictions = model.predict(tr)
print('Predictions')
print(predictions)
#test_targets=np.full(len(predictions),1)
# Calculate the mean squared error (MSE) for each prediction
mse = np.mean(np.square(predictions - tr), axis=1)
print(mse)
mme=np.mean(np.square(predictions-tr))
print(mme)
# Define a threshold for anomaly detection
threshold = 1.1
#model.save('D:/modelnew1.h5')
#test_targets
#loss, accuracy = model.evaluate(test_data, test_targets)
# Save model with specified format (e.g., SavedModel format)
#tf.saved_model.save(model, 'path/to/save/model/saved_model')
#train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
# Identify anomalies based on the threshold
#tl=pd.read_csv('D:/test_route3.csv')

anomalies = np.where(mse <= threshold)[0]
print(len(anomalies),len(tr))
print(mse)
#print((len(anomalies)/len(data))*100)
#test_labels=tl['test']
#test_labels[test_labels == -1] = 1
#test_labels[test_labels == 1] = 0
#test_labels=np.full(len(ft),1)
# Print the indices of the detected anomalies
print("Detected anomalies at indices:", anomalies)
binary_predictions = np.where(mse <= threshold, 1, 0)
print(np.where(binary_predictions==0))
#test_labels=np.full(len(predictions),0)
# Calculate accuracy
accuracy = accuracy_score(hl, binary_predictions)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(h, binary_predictions)
print("Precision:", precision)

# Calculate recall
recall = recall_score(h1, binary_predictions,zero_division=1)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(h1, binary_predictions,zero_division=1)
print("F1-score:", f1)

"""


