# MLOps-Project
# Pneumonia Onset Prediction in ICU Patients
# Overview
This repository contains code to build a machine learning model that can predict the onset of pneumonia in ICU patients 48 hours before the actual diagnosis. The model is trained on the freely available eICU critical care dataset.

## Prerequisites

To build, test, and deploy this project, you will need:

- **Airflow** to schedule and run pipelines
- **Python** to code pipelines and data science
- **Docker** to package and containerize
- **DVC** to version data and ML models
- **Git/GitHub** for source code management

### Airflow

- Install Airflow
- Know how to write Airflow DAGs
- Run Airflow server

### Python  

- Install Python 3.7+
- Import libraries like Pandas, Numpy  

### Docker

- Install Docker engine  

### DVC

- Install DVC
- Configure DVC remote storage   
- Manage data and models with DVC

### Git/GitHub  

- Clone repo
- Commit code 
- Push code

This covers the core tools and knowledge needed. The tools will be used together to build, test, and deploy the data pipelines.

# Download Data
Automated data download

## Data
The eICU critical care dataset is hosted by MIT Laboratory for Computational Physiology and can be accessed after signing a DUA and completing required CITI training on human subjects research. The data contains over 200k admissions across multiple ICUs in the US.

The features used include demographics, vital signs, and lab test results. The target variable is a diagnosis of pneumonia based on ICD-9 codes.

This dataset contains deidentified health data from intensive care unit (ICU) patients. We have 46 features and 

The data includes:

- Vital signs: `temperature`, `respiration`, `sao2`, `heartrate`, etc.

- Lab results: `WBC`, `bicarbonate`, `glucose`, etc.

- Aggregated measurements: `temp_mean`, `respiration_stdev`, etc. 

- Medications

- APACHE components

- Care plans

- Admission diagnoses 

- Patient history

- Time-stamped problem list diagnoses

- Chosen treatments

The data is collected from different care units into a common warehouse. Each unit provides data through specific interfaces that transform and load the data. Units may have different interfaces in place. 

The lack of an interface for a unit will result in missing data for patients from that unit. The data is provided as relational database tables joined by keys.

This dataset can be used to train machine learning models to predict pneumonia diagnoses based on the vital signs, lab results, and other clinical measurements provided.

## Features  

**Patient Information**

- `patientunitstayid`: Unique identifier for each hospital stay
- `gender`: Biological sex    
- `age`: Age in years  
- `ethnicity`: Racial/ethnic background

**Admission Information**  

- `admissionheight`: Height upon admission in cm   
- `admissionweight`: Weight upon admission in kg

**Time Information**   

- `observationoffset`: Timestep index for observation data  
- `offsettime`: Time in hours elapsed since admission
- `labresultoffset`: Timestep index for lab test results 
- `timediff`: Time difference between observations in hours

**Vital Signs**  

- `temperature`: Body temperature in Celsius   
- `sao2`: Blood oxygen saturation percentage  
- `heartrate`: Heart rate in beats per minute
- `respiration`: Breathing rate in breaths per minute
- `systemicsystolic`: Systolic arterial pressure in mmHg
- `systemicdiastolic`: Diastolic arterial pressure in mmHg 

**Vital Sign Averages**   

- `temp_mean`: Average temperature across stay
- `temp_stdev`: Standard deviation of temperature across stay  
- `sao2_mean`: Average oxygen saturation across stay
- `sao2_stdev`: Standard deviation of oxygen saturation across stay  
- `heartrate_mean`: Average heart rate across stay 
- `heartrate_stdev`: Standard deviation of heart rate across stay
- `respiration_mean`: Average respiration rate across stay 
- `respiration_stdev`: Standard deviation of respiration rate across stay 
- `systemicsystolic_mean`: Average systolic pressure across stay
- `systemicsystolic_stdev`: Standard deviation of systolic pressure across stay 
- `systemicdiastolic_mean`: Average diastolic pressure across stay  
- `systemicdiastolic_stdev`: Standard deviation of diastolic pressure across stay

**Lab Test Results**
  
- `BUN`: Blood urea nitrogen level in mg/dL   
- `Hct`: Hematocrit percentage 
- `Hgb`: Hemoglobin level in g/dL    
- `MCH`: Mean corpuscular hemoglobin in pg  
- `MCHC`: Mean corpuscular hemoglobin concentration in g/dL
- `MCV`: Mean corpuscular volume in fL  
- `RBC`: Red blood cell count in million cells/mcL   
- `RDW`: Red cell distribution width percentage  
- `WBC`: White blood cell count in thousands/mcL    
- `bicarbonate`: Bicarbonate level in mmol/L
- `calcium`: Calcium level in mg/dL  
- `chloride`: Chloride level in mmol/L   
- `creatinine`: Creatinine level in mg/dL  
- `glucose`: Glucose level in mg/dL  
- `platelets`: Platelet count in thousands/mcL 
- `potassium`: Potassium level in mmol/L   
- `sodium`: Sodium level in mmol/L

# Installing
### 1. Clone the repository: 
~~~
git clone https://github.com/bsr11272/MLOps-Project.git
~~~
### 2. Setup Virtual Environment
~~~
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\Activate
~~~

### 3. Installation

To install the dependencies, run:

~~~
pip install -r requirements.txt
~~~

### 4. Docker:
~~~
echo -e "AIRFLOW_UID=$(id -u)" > .env
echo "AIRFLOW_HOME_DIR=$(pwd)" >> .env

docker compose up airflow-init
docker compose up
~~~

## Preprocessing 

**Step 1: Concatenate 15 files**

This step involves combining the data from 15 individual files into a single file. This can be useful for aggregating data from multiple sources, or for preparing data for further analysis.

**Step 2: Pivot Lab files**

This step involves transforming the data from the concatenated file into a pivot table format. This can be useful for summarizing and analyzing the data, or for identifying trends and patterns.

**Step 3: Fetch sorted labs and divide by group by patientunitstayID and sort the values by lab result offset then use customfill functions ffill and bfill if offsettime difference is less than 24 hrs**

This step involves sorting the lab results for each patient by offset time and filling in any missing values using the `ffill()` and `bfill()` functions.

**Step 4: Merge labs and Vitals**

This step involves merging the lab results with the vital signs data. This can be useful for creating a single DataFrame that contains all of the relevant data for each patient.

**Step 5: Drop duplicates**

This step involves removing duplicate rows from the DataFrame. This can be useful for ensuring that the DataFrame contains unique data for each patient.

**Step 6: Convert List of Strings to Float Values**

This step involves converting the lab results from list of strings to floats. This can be useful for performing mathematical operations on the data, or for creating visualizations.

# Results
The best performing model are in progress on the test set. The most important features were found to be respiratory rate, WBC count, and temperature.

# Contributing
Contributions to improve the model performance or implementation are welcome! Please open an issue or PR.

# License
The eICU dataset is subject to DUA restrictions. All other code in this repository is under the MIT license.
