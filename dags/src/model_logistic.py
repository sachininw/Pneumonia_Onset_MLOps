"""Split train and test"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import logging
import sys
import warnings
from urllib.parse import urlparse
from sklearn.linear_model import LogisticRegression
from pathlib import Path


import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



def eval_metrics(model, Y_test, y_pred, X_test_scaled):
    # Calculate and print accuracy on the test set
    accuracy = accuracy_score(Y_test, y_pred)

    # Calculate and print ROC AUC on the test set
    roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test_scaled)[:, 1])

    # Calculate and print precision on the test set
    precision = precision_score(Y_test, y_pred)

    # Calculate and print recall on the test set
    recall = recall_score(Y_test, y_pred)
    
    # Calculate and print recall on the test set
    f1_score_ = f1_score(Y_test, y_pred)
    
    return accuracy, roc_auc, precision, recall, f1_score_



def Logistic_scaling(X_train, X_test):
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

def Logistic_preprocessing(data):
    
    
    data = data.drop(columns=['patientunitstayid','observationoffset','offsettime', "labresultoffset", "Relative_time"], axis = 1)
    
    #Dropna 
    data = data.dropna()
    ##Label encoding
    data = pd.get_dummies(data, columns=['gender','ethnicity'])
    return data
    

def Logistic_model(data_path, mode = 'train'):
    
    mlflow.set_experiment("Logistic_model")
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    data = pd.read_csv(data_path)
    data = Logistic_preprocessing(data)
    print("data.shape", data.shape)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["y"], axis=1)
    test_x = test.drop(["y"], axis=1)
    train_y = train[["y"]]
    test_y = test[["y"]]
    
    train_x, test_x = Logistic_scaling(train_x, test_x)

    C = 0.5
    l1_ratio = 0.5

    with mlflow.start_run():
        
        lr = LogisticRegression(penalty='elasticnet' ,C=C, l1_ratio=l1_ratio, random_state=42, solver='saga')
        lr.fit(train_x, train_y)
        y_pred = lr.predict(test_x)

        (accuracy, roc_auc, precision, recall, f1_score_) = eval_metrics(lr, test_y, y_pred, test_x)

        mlflow.log_param("C", C)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1_score_)

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="LogisticRegressionModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)
        
        #mlflow.log_artifacts('artifacts')
            