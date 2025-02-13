import pandas as pd
import re
import pickle
import os
from pathlib import Path


def load_data(label):
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """
    
    
    #df = pd.read_csv(os.path.join("\\".join(cd.split("\\")[:-2]), "./data/pneumonia_converted_data.csv"))
    if label == 'pneumonia':
        path = Path(__file__).parent.parent.parent/'data'/'Initial_data'/'pneumonia_data.csv'
    elif label == 'nopneumonia':
        path = Path(__file__).parent.parent.parent/'data'/'Initial_data'/'noPneumonia_data.csv'
    df = pd.read_csv(path)
    serialized_data = pickle.dumps(df)
    
    return serialized_data


def age_preprocessing(label, data):

    """
    Deserializes data, performs data preprocessing, and returns serialized clustered data.

    Args:
        data (bytes): Serialized data to be deserialized and processed.

    Returns:
        bytes: Serialized clustered data.
    """

    def str_to_int(age):
        numbers = re.findall('\d+', str(age))
        return int(numbers[0])

    data['age'] = data['age'].apply(str_to_int)
    
    output_directory = Path(__file__).parent.parent.parent/'data'/'Initial_data'    
    if label == 'pneumonia':
        output_file_path = os.path.join(output_directory, 'Pneumonia_data_age_preprocessed.csv')
        data.to_csv(output_file_path, index=False)

    elif label == 'nopneumonia':
        output_file_path = os.path.join(output_directory, 'noPneumonia_data_age_preprocessed.csv')
        data.to_csv(output_file_path, index=False)