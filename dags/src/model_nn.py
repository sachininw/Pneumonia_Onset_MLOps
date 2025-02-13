import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import logging
import sys
import warnings
from urllib.parse import urlparse
from sklearn.linear_model import LogisticRegression


import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



# Finally, training with the best hyperparameters and saving the model
# ... [your existing final training and model saving code] ...
##Training with the best hparam set
def nn_scaling(X_train, X_test):
    scaler = StandardScaler()

    # Fit the scaler on the reshaped array and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    ##Scaling test

    #Record the scaling statistics
    mean_values = scaler.mean_
    std_values = np.sqrt(scaler.var_)

    # Display the scaling statistics
    #print("Mean values for each feature:", mean_values)
    #print("Standard deviation values for each feature:", std_values)

    # Transform the test array using the scaler
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

def nn_preprocessing(data):
    
    
    data = data.drop(columns=['patientunitstayid','observationoffset','offsettime', "labresultoffset", "Relative_time"], axis = 1)
    
    #Dropna 
    data = data.dropna()
    ##Label encoding
    data = pd.get_dummies(data, columns=['gender','ethnicity'])
    return data


def best_nn_model(data_path):
    
    def train_test_model(X_train, Y_train, X_test, Y_test, hparams, epochs):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
        ])
        model.compile(
            optimizer=hparams[HP_OPTIMIZER],
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        #mlflow.log_params(hparams)
        model.fit(X_train, Y_train, epochs=epochs, callbacks=[
            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
            hp.KerasCallback(logdir, hparams),  # log hparams
        ]) # Run with 1 epoch to speed things up for demo purposes
        _,accuracy = model.evaluate(X_test, Y_test)
        mlflow.log_metric("METRIC_ACCURACY", accuracy)
        return model, accuracy
        
        
        # Adjusted run function with MLflow
    def run(run_dir, X_train, Y_train, X_test, Y_test, hparams, epochs, max_accuracy, best_hyperparameters):
        with mlflow.start_run():
            model, accuracy = train_test_model(X_train, Y_train, X_test, Y_test, hparams, epochs)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_hyperparameters = hparams
            predictions = model.predict(X_train)
            signature = infer_signature(X_train, predictions)
            mlflow.tensorflow.log_model(model = model, artifact_path="model", signature=signature)
        return max_accuracy, best_hyperparameters
        
    # Setup MLflow experiment
    mlflow.set_experiment("tensorflow_hyperparam_tuning")
    warnings.filterwarnings("ignore")
        
    # Define your hyperparameters and metrics as before
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([8, 16]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    METRIC_ACCURACY = 'accuracy'
    best_hyperparameters = None
    max_accuracy = 0.0
    
    logdir = ''
    
    data = pd.read_csv(data_path)
    data = nn_preprocessing(data)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    # The predicted column is "quality" which is a scalar from [3, 9]
    X_train = train.drop(["y"], axis=1)
    X_test = test.drop(["y"], axis=1)
    Y_train = train[["y"]]
    Y_test = test[["y"]]
    
    X_train, X_test = nn_scaling(X_train, X_test)
    
    
    epochs = 1
    session_num = 0
    # ... [rest of your hyperparameter tuning loop] ...
    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                    HP_OPTIMIZER: optimizer,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
            
                max_accuracy,best_hyperparameters = run('logs/hparam_tuning/' + run_name, X_train, Y_train, X_test, Y_test, hparams, epochs, max_accuracy, best_hyperparameters)
                session_num += 1

    print('Best Hyperparameters:', best_hyperparameters)
    print('Maximum Accuracy:', max_accuracy)
    
    
    
    epochs = 10
    model, accuracy = train_test_model(X_train, Y_train, X_test, Y_test, best_hyperparameters, epochs)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    model.save('models/HPtuningNNModel.h5')