import pandas as pd
from pathlib import Path
import os


def read_concatenate_save():

    input_directory = Path(__file__).parent.parent.parent/'data'/'Pneumonia_Dataset'
    output_directory = Path(__file__).parent.parent.parent/'data'/'labs_vitals_data'

    output_directory.mkdir(parents=True, exist_ok=True)

    pneumonia_labs = []

    pneumonia_vitals = []

    no_pneumonia_labs = []

    no_pneumonia_vitals = []
 
    # List all files in the input directory

    for file in os.listdir(input_directory):

        if file.endswith('.csv'):  # Check if the file is a CSV file

            file_path = os.path.join(input_directory, file)

            if 'noPneumoniaLabs' in file:

                no_pneumonia_labs.append(file_path)

            elif 'noPneumoniaVitals' in file:

                no_pneumonia_vitals.append(file_path)

            elif 'pneumoniaLabs' in file:

                pneumonia_labs.append(file_path)

            elif 'PneumoniaVitals' in file:

                pneumonia_vitals.append(file_path)
 
    # Function to concatenate files in each category

    def concatenate_files(file_list, category):

        if file_list:  # Check if file list is not empty

            dataframes = [pd.read_csv(f) for f in file_list]

            concatenated_df = pd.concat(dataframes, ignore_index=True)

            output_path = os.path.join(output_directory, f'{category}.csv')

            concatenated_df.to_csv(output_path, index=False)

            print(f'{category} data saved to {output_path}')

        else:

            print(f"No files found for {category}")
 
    # Concatenate and save files in each category

    concatenate_files(pneumonia_labs, 'pneumoniaLabs')

    concatenate_files(pneumonia_vitals, 'PneumoniaVitals')

    concatenate_files(no_pneumonia_labs, 'noPneumoniaLabs')

    concatenate_files(no_pneumonia_vitals, 'noPneumoniaVitals')