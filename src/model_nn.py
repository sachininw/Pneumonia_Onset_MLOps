import datetime
import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature
import logging

def preprocess_data(data_path):
    try:
        data = pd.read_csv(data_path)
        data = data.drop(columns=['patientunitstayid', 'observationoffset', 'offsettime', "labresultoffset", "Relative_time"], axis=1)
        data = data.dropna()
        data = pd.get_dummies(data, columns=['gender', 'ethnicity'])
        return train_test_split(data, test_size=0.2, random_state=42)
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        raise

def scale_data(X_train, X_test):
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    except Exception as e:
        logging.error(f"Data scaling failed: {e}")
        raise

def build_model(num_units, dropout, optimizer):
    try:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_units, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
        ])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        logging.error(f"Model building failed: {e}")
        raise

def train_model(model, X_train, Y_train, X_test, Y_test, epochs, logdir, hparams):
    try:
        model.fit(X_train, Y_train, epochs=epochs, callbacks=[tf.keras.callbacks.TensorBoard(logdir), hp.KerasCallback(logdir, hparams)])
        return model.evaluate(X_test, Y_test)[1]
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def log_model(tf_model, X_train, accuracy, artifact_path="model"):
    try:
        predictions = tf_model.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.tensorflow.log_model(tf_model, artifact_path=artifact_path, signature=signature)
        mlflow.log_metric("accuracy", accuracy)
    except Exception as e:
        logging.error(f"Model logging failed: {e}")

def run_best_model(data_path, logdir):
    try:
        # Contingency plan for empty logdir
        if not logdir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            default_logdir = f'logs/default_run_{timestamp}'
            os.makedirs(default_logdir, exist_ok=True)
            logdir = default_logdir
            logging.info(f"No logdir provided. Using default logdir: {logdir}")
        train, test = preprocess_data(data_path)
        X_train, X_test = train.drop(["y"], axis=1), test.drop(["y"], axis=1)
        Y_train, Y_test = train[["y"]], test[["y"]]
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

        # Define hyperparameters
        hp_units = hp.HParam('num_units', hp.Discrete([4, 8]))
        hp_dropout = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
        hp_optimizer = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
        metric_accuracy = 'accuracy'

        session_num = 0
        best_accuracy = 0.0
        best_hparams = None
        for num_units in hp_units.domain.values:
            for dropout in [hp_dropout.domain.min_value, hp_dropout.domain.max_value]:
                for optimizer in hp_optimizer.domain.values:
                    hparams = {hp_units: num_units, hp_dropout: dropout, hp_optimizer: optimizer}
                    model = build_model(num_units, dropout, optimizer)
                    with mlflow.start_run():
                        accuracy = train_model(model, X_train_scaled, Y_train, X_test_scaled, Y_test, epochs=1, logdir=f'{logdir}/run-{session_num}', hparams=hparams)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_hparams = hparams
                        log_model(model, X_train_scaled, accuracy, artifact_path=f"model_run_{session_num}")
                    session_num+=1


        logging.info(f'Best Hyperparameters: {best_hparams}')
        logging.info(f'Maximum Accuracy: {best_accuracy}')

        # Train and save the final model with best hyperparameters
        final_model = build_model(best_hparams[hp_units], best_hparams[hp_dropout], best_hparams[hp_optimizer])
        final_accuracy = train_model(final_model, X_train_scaled, Y_train, X_test_scaled, Y_test, epochs= 5, logdir=logdir, hparams=best_hparams)
        final_model.save('models/HPtuningNNModel.h5')
    except Exception as e:
        logging.error(f"run_best_model failed: {e}")
        raise

# Example usage
# run_best_model(data_path='path_to_your_data.csv', logdir='path_to_log_directory')