import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Start an MLflow run
mlflow.start_run()

# Log the parameters used for training
mlflow.log_param("model_type", "LSTM")
mlflow.log_param("hidden_units", "64")
mlflow.log_param("optimizer", "adam")
mlflow.log_param("loss_function", "mse")
mlflow.log_param("epochs", 12)

# Define your LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(None, 1), return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Load and preprocess the training data
tr = pd.read_csv('route_github.csv')
vessel = ['T.ADALYN','DALI','NAVE GALACTIC','STARDUST','ISLENO','ARRIBA','BEVERLY M I','CHICAGO EXPRESS','ELKA HERCULES','FROSTI','GLOVIS CAPTAIN','GOLDEN ALASKA','HYUNDAI PRIDE','KEN HOU','RUBY M','KODIAK ENTERPRISE','VIDA','WESTWOOD RAINIER']
tr = tr[tr['VesselName'].isin(vessel)]
tr['BaseDateTime'] = tr['BaseDateTime'].str.replace('T', ' ')
tr = tr.dropna(axis=0)
tr = tr.drop(['VesselName','IMO','SOG','COG','CallSign','VesselType','Status','Length','Width','Draft','Cargo','TransceiverClass','BaseDateTime'], axis=1)

scaler = StandardScaler()
normalized_data = scaler.fit_transform(tr)

# Log the training data shape
mlflow.log_param("num_samples", len(tr))
mlflow.log_param("num_features", tr.shape[1])

# Generate the labels (assuming `e` contains the labels)
e = np.full(len(normalized_data), 0)

# Train the model
model.fit(normalized_data, e, epochs=12)

# Log the trained model
mlflow.tensorflow.log_model(model, "modelroute")

# End the MLflow run
mlflow.end_run()
