import ast
import os
from pathlib import Path
import pickle
 
    # Function to convert a list of strings to floats and merge duplicates
 
def convert_to_float_and_merge(item):
        try:
            item_list = ast.literal_eval(item)  # Convert the string to a list
            if isinstance(item_list, list):
                float_values = [float(val) for val in item_list if isinstance(val, (int, float)) or val.replace(".", "", 1).isdigit()]
                unique_float_values = list(set(float_values))
                return sum(unique_float_values)  # Merge duplicate values by summing
        except (ValueError, SyntaxError):
            pass  # Handle non-list values
        return item  # Return the original item for non-list values
 
    # Apply the conversion function to each element in the DataFrame
def apply_conversion(label, df):
    output_directory = Path(__file__).parent.parent.parent/'data'/'Initial_data'
    df = pickle.loads(df)
    cols = ['BUN','Hct','Hgb','MCH','MCHC','MCV','RBC','RDW','WBC x 1000','bicarbonate','calcium','chloride','creatinine','glucose','platelets x 1000','potassium','sodium']
    for col in cols:
        df[col] = df[col].apply(convert_to_float_and_merge)
    if label == 'pneumonia':
        output_file_path = os.path.join(output_directory, 'Pneumonia_data.csv')
        df.to_csv(output_file_path, index=False)