import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
 
    # Function to convert a list of strings to floats and merge duplicates
 
def label_creation_apply(label):
    output_directory = Path(__file__).parent.parent.parent/'data'/'Initial_data'
    
    if label == 'pneumonia':
        df = df = pd.read_csv(output_directory/'pneumonia_data.csv')
        df["Relative_time"] = df["offsettime"] - np.minimum(df["observationoffset"], df["labresultoffset"])
        df['y'] = np.where(df['Relative_time'] > 0, 1, 0)
        output_file_path = os.path.join(output_directory, 'Pneumonia_data.csv')
        df.to_csv(output_file_path, index=False)
    elif label == 'nopneumonia':
        df = pd.read_csv(output_directory/'nopneumonia_data.csv')
        df['y'] = 0
        output_file_path = os.path.join(output_directory, 'noPneumonia_data.csv')
        df.to_csv(output_file_path, index=False)