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

- Vital signs

- Lab results

- Aggregated measurements

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


# License
The eICU dataset is subject to DUA restrictions. All other code in this repository is under the MIT license.
