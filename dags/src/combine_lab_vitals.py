import numpy as np
import pandas as pd
from pathlib import Path
import math
from typing import List
import os

#Fetch grouped labs for each patient and sort lab results in the ascending order of result offset time 
def fetch_grouped_sorted_labs(label) -> pd.DataFrame:


  input_directory = Path(__file__).parent.parent.parent/'data'
  if label == 'pneumonia':
    df = pd.read_csv(input_directory/'structured_data'/'PneumoniaLabs.csv')
  elif label == 'nopneumonia':
    df = pd.read_csv(input_directory/'structured_data'/'noPneumoniaLabs.csv')

  sorted_grouped_labs = df.groupby('patientunitstayid').apply(lambda x: x.sort_values(by='labresultoffset', ascending=True))
  sorted_grouped_labs.reset_index(drop=True, inplace=True)

  return sorted_grouped_labs


#Foreward fill and backward fill the mising lab values if the result was taken within 24hrs.
#Create custom fill

#Perform custom fill on each patient
def fetch_filled_labs(df: pd.DataFrame) -> pd.DataFrame:


  def custom_fill(group):

    condition = (group['labresultoffset'].diff() <= 1440) & group['labresultoffset'].notna()
    group.loc[condition] = group.loc[condition].ffill()
    group.loc[condition] = group.loc[condition].bfill()

    return group

  filled_labs = df.groupby('patientunitstayid').apply(custom_fill)

  return filled_labs.reset_index(drop=True)

#Fetch merged vital and lab data
def fetch_merged_vitals_labs(label, labs: pd.DataFrame) -> pd.DataFrame:
  input_directory = Path(__file__).parent.parent.parent/'data'
  if label == 'pneumonia':
    vitals = pd.read_csv(input_directory/'labs_vitals_data'/'PneumoniaVitals.csv')
  elif label == 'nopneumonia':
    vitals = pd.read_csv(input_directory/'labs_vitals_data'/'noPneumoniaVitals.csv')
    vitals = vitals.sample(frac=0.05)

  merged_vitals_labs = pd.merge(vitals, labs, on='patientunitstayid', how='inner')

  return merged_vitals_labs

#Fetch lab-vital combinations that occured within 24hrs of each other's offset times
def fetch_closeby_vitalslabs(df: pd.DataFrame) -> pd.DataFrame:

  closeby_vitalslabs = df[abs(df['observationoffset'] - df['labresultoffset']) <= 1440]

  return closeby_vitalslabs

#Drop duplicates
def dropduplicates(label,df: pd.DataFrame) -> pd.DataFrame:
  
    df_droped_duplicates = df.drop_duplicates()

    output_directory = Path(__file__).parent.parent.parent/'data'/'Initial_data'

    output_directory.mkdir(parents=True, exist_ok=True)

    if label == 'pneumonia':
      output_file_path = os.path.join(output_directory, 'Pneumonia_data.csv')
      df_droped_duplicates.to_csv(output_file_path, index=False)
    
    elif label == 'nopneumonia':
      output_file_path = os.path.join(output_directory, 'noPneumonia_data.csv')
      df_droped_duplicates.to_csv(output_file_path, index=False)


