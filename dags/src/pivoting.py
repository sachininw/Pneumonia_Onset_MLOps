import pandas as pd
from pathlib import Path
import os
 


def pivot_lab_results_and_save(lab_identifier='Labs'):

    input_directory = Path(__file__).parent.parent.parent/'data'/'labs_vitals_data'

    output_directory = Path(__file__).parent.parent.parent/'data'/'structured_data'
    output_directory.mkdir(parents=True, exist_ok=True)

    for file in os.listdir(input_directory):

        if file.endswith('.csv') and lab_identifier in file:

            file_path = os.path.join(input_directory, file)

            df = pd.read_csv(file_path)
 
            # Check if required columns are in the DataFrame

            required_columns = ['patientunitstayid', 'labresultoffset', 'labname', 'labresulttext']

            if all(column in df.columns for column in required_columns):

                try:

                    # Pivot the table using groupby and unstack

                    pivoted_data = df.groupby(['patientunitstayid', 'labresultoffset', 'labname'])['labresulttext'].apply(list).unstack(fill_value='')

                    pivoted_data.reset_index(inplace=True)
 
                    # Determine the output file name based on the input file

                    output_file_name = 'noPneumoniaLabs.csv' if 'Pneumonia' in file else 'PneumoniaLabs.csv'
 
                    # Save the pivoted DataFrame to a CSV file

                    output_file_path = os.path.join(output_directory, output_file_name)

                    pivoted_data.to_csv(output_file_path, index=False)

                    print(f"Pivoted lab data from {file} saved to {output_file_path}")
 
                except Exception as e:

                    print(f"An error occurred while processing {file}: {e}")
 
            else:

                print(f"File {file} skipped: required columns not found")
 