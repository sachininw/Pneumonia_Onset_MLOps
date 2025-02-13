import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_test_split_function():
    path = Path(__file__).parent.parent.parent/'data'/'Initial_data'
    df_pneumonia = pd.read_csv(os.path.join(path, 'Pneumonia_data_age_preprocessed.csv'))
    df_nopneumonia = pd.read_csv(os.path.join(path, 'noPneumonia_data_age_preprocessed.csv'))
    
    
    
    combined_df = pd.concat([df_pneumonia.sample(n=min(df_pneumonia.shape[0],df_nopneumonia.shape[0]), random_state=42), df_nopneumonia.sample(n=min(df_pneumonia.shape[0],df_nopneumonia.shape[0]), random_state=42)], ignore_index=True)
    
    train, test = train_test_split(combined_df, test_size=0.2, random_state=42)
    
    output_directory_train = Path(__file__).parent.parent.parent/'working_data'/'TRAIN'
    output_directory_train.mkdir(parents=True, exist_ok=True)
    output_file_path = os.path.join(output_directory_train, 'train.csv')
    train.to_csv(output_file_path, index=False)
    
    
    output_directory_test = Path(__file__).parent.parent.parent/'working_data'/'TEST'
    output_directory_test.mkdir(parents=True, exist_ok=True)
    output_file_path = os.path.join(output_directory_test, 'test.csv')
    test.to_csv(output_file_path, index=False)