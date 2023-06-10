import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import mlflow

default_args = {
    'owner': 'your_name',
    'start_date': datetime.datetime(2023, 6, 1),
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}

dag = DAG('mlflow_training_dag', default_args=default_args, schedule_interval='0 0 * * *')

def train_model():
    import mlflow
    import mlflow.tensorflow
    import numpy as np
    import tensorflow as tf
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
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
   
    # Train the model
    model.fit(normalized_data, e, epochs=12)

    # Log the trained model
    mlflow.tensorflow.log_model(model, "modelroute")

    # End the MLflow run
    mlflow.end_run()

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    dag=dag
)

train_model_task
