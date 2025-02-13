import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature

def scale_and_preprocess_data(data):
    # Combine preprocessing steps
    data = data.drop(columns=['patientunitstayid', 'observationoffset', 'offsettime', 'labresultoffset', 'Relative_time'])
    data = pd.get_dummies(data, columns=['gender', 'ethnicity'])

    return data

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_and_log_model(model, X_train, X_test, y_train, y_test):
    # Train and log the model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    signature = infer_signature(X_train, predictions)

    with mlflow.start_run():
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model", signature=signature)

def random_forest_pipeline(data_path):
    data = pd.read_csv(data_path)
    data = scale_and_preprocess_data(data)

    X = data.drop(["y"], axis=1)
    y = data["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = scale_features(X_train, X_test)

    hyperparameters = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.01, 0.1]  # for XGBoost
    }

    for n_estimators in hyperparameters['n_estimators']:
        for max_depth in hyperparameters['max_depth']:
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            train_and_log_model(rf_model, X_train, X_test, y_train, y_test)

            for learning_rate in hyperparameters['learning_rate']:
                xgb_model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
                train_and_log_model(xgb_model, X_train, X_test, y_train, y_test)