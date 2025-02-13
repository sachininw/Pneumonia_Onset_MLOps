# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.docker_operator import DockerOperator
from datetime import datetime, timedelta

from src.preprocessing import age_preprocessing, load_data
from src.labs_vitals_generator import read_concatenate_save
from src.pivoting import pivot_lab_results_and_save
from src.combine_lab_vitals import fetch_grouped_sorted_labs, fetch_filled_labs, fetch_merged_vitals_labs, fetch_closeby_vitalslabs, dropduplicates
from src.data_download import download
from src.cleaning import apply_conversion
from src.label_creation import label_creation_apply
from src.test_train_split import train_test_split_function
from src.model_Logistic import Logistic_model
from src.model_nn_new import run_best_model
from src.model_rf import random_forest_pipeline

import os
from pathlib import Path
from docker.types import Mount
from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')



# Define default arguments for your DAG
default_args = {
    'owner': 'Sai',
    'start_date': datetime(2023, 9, 17),
    'retries': 3, # Number of retries in case of task failure
    'retry_delay': timedelta(seconds=5), # Delay before retries
}


# Create a DAG instance named 'your_python_dag' with the defined default arguments
dag = DAG(
    'your_python_dag',
    default_args=default_args,
    description='Your Python DAG Description',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)

data_download_task = PythonOperator(
    task_id='data_download_task',
    python_callable=download,
    dag=dag,
)
# Define PythonOperators for each function
lab_vital_generator_task = PythonOperator(
    task_id='lab_vital_generator_task',
    python_callable=read_concatenate_save,
    dag=dag,
)

pivoting_task = PythonOperator(
    task_id='pivoting_task',
    python_callable=pivot_lab_results_and_save,
    dag=dag,
)



fetch_grouped_sorted_labs_pneumonia = PythonOperator(
    task_id='fetch_grouped_sorted_labs_pneumonia',
    python_callable=fetch_grouped_sorted_labs,
    op_args=['pneumonia'],
    dag=dag,
)
fetch_grouped_sorted_labs_nopneumonia = PythonOperator(
    task_id='fetch_grouped_sorted_labs_nopneumonia',
    python_callable=fetch_grouped_sorted_labs,
    op_args=['nopneumonia'],
    dag=dag,
)


fetch_filled_labs_pneumonia = PythonOperator(
    task_id='fetch_filled_labs_pneumonia',
    python_callable=fetch_filled_labs,
    op_args=[fetch_grouped_sorted_labs_pneumonia.output],
    dag=dag,
)
fetch_filled_labs_nopneumonia = PythonOperator(
    task_id='fetch_filled_labs_nopneumonia',
    python_callable=fetch_filled_labs,
    op_args=[fetch_grouped_sorted_labs_nopneumonia.output],
    dag=dag,
)

fetch_merged_vitals_labs_pneumonia = PythonOperator(
    task_id='fetch_merged_vitals_labs_pneumonia',
    python_callable=fetch_merged_vitals_labs,
    op_args=['pneumonia', fetch_filled_labs_pneumonia.output],
    dag=dag,
)
fetch_merged_vitals_labs_nopneumonia = PythonOperator(
    task_id='fetch_merged_vitals_labs_nopneumonia',
    python_callable=fetch_merged_vitals_labs,
    op_args=['nopneumonia', fetch_filled_labs_nopneumonia.output],
    dag=dag,
)


fetch_closeby_vitalslabs_pneumonia = PythonOperator(
    task_id='fetch_closeby_vitalslabs_pneumonia',
    python_callable=fetch_closeby_vitalslabs,
    op_args=[fetch_merged_vitals_labs_pneumonia.output],
    dag=dag,
)
fetch_closeby_vitalslabs_nopneumonia = PythonOperator(
    task_id='fetch_closeby_vitalslabs_nopneumonia',
    python_callable=fetch_closeby_vitalslabs,
    op_args=[fetch_merged_vitals_labs_nopneumonia.output],
    dag=dag,
)


dropduplicates_pneumonia = PythonOperator(
    task_id='dropduplicates_pneumonia',
    python_callable=dropduplicates,
    op_args=['pneumonia', fetch_closeby_vitalslabs_pneumonia.output],
    dag=dag,
)
dropduplicates_nopneumonia = PythonOperator(
    task_id='dropduplicates_nopneumonia',
    python_callable=dropduplicates,
    op_args=['nopneumonia', fetch_closeby_vitalslabs_nopneumonia.output],
    dag=dag,
)


# Task to load data, calls the 'load_data' Python function
load_pneumonia_data_task = PythonOperator(
    task_id='load_pneumonia_data_task',
    python_callable=load_data,
    op_args=['pneumonia'],
    dag=dag,
)
load_nopneumonia_data_task = PythonOperator(
    task_id='load_nopneumonia_data_task',
    python_callable=load_data,
    op_args=['nopneumonia'],
    dag=dag,
)

cleaning_task_pneumonia = PythonOperator(
    task_id='cleaning_task_pneumonia',
    python_callable=apply_conversion,
    op_args=['pneumonia', load_pneumonia_data_task.output],
    dag=dag,
)
cleaning_task_nopneumonia = PythonOperator(
    task_id='cleaning_task_nopneumonia',
    python_callable=apply_conversion,
    op_args=['nopneumonia', load_nopneumonia_data_task.output],
    dag=dag,
)


label_creation_task_pneumonia = PythonOperator(
    task_id='label_creation_task_pneumonia',
    python_callable=label_creation_apply,
    op_args=['pneumonia'],
    dag=dag,
)
label_creation_task_nopneumonia = PythonOperator(
    task_id='label_creation_task_nopneumonia',
    python_callable=label_creation_apply,
    op_args=['nopneumonia'],
    dag=dag,
)

age_preprocessing_task_pneumonia = PythonOperator(
    task_id='age_preprocessing_task_pneumonia',
    python_callable=age_preprocessing,
    op_args=['pneumonia', label_creation_task_pneumonia.output],
    dag=dag,
)
age_preprocessing_task_nopneumonia = PythonOperator(
    task_id='age_preprocessing_task_nopneumonia',
    python_callable=age_preprocessing,
    op_args=['nopneumonia', label_creation_task_nopneumonia.output],
    dag=dag,
)

### Tf DAG
'''
def my_docker_operator(
    task_id,
    image,
    command,
    mounts,
    dag,
    network_mode="bridge",
    privileged=False,
    # Add this line to fix DockerOperator permission denied airflow dag
    owner="airflow",
    group="airflow",
):

    return DockerOperator(
        task_id=task_id,
        image=image,
        command=command,
        mounts = mounts,
        network_mode=network_mode,
        privileged=privileged,
        owner=owner,
        group=group,
        dag=dag
    )

'''
'''
#path = Path(__file__).parent/'src'/'TFX'
path = 'C:/Users/saire/Desktop/School/MLOps/Project/MLOps_Project/dags/src/TFX'
tfx_schema_task = DockerOperator(
    task_id = 'tfx_schema_gen',
    image='tensorflow/tensorflow:latest-gpu',  # Use your TensorFlow Docker image
    command='python -c "import sys; sys.path.append(\'/code\'); import tfx; tfx.schema_gen()"',  # Command to run your script
    mounts=[
        Mount(
            source= str(path), 
            target='/code',
            type='bind'
            )
        ],  # Mount the src directory to /code in the container
    network_mode="bridge",
    docker_url="unix://var/run/docker.sock",
    dag=dag,
)
'''


train_test_split_task = PythonOperator(
    task_id='train_test_split_task',
    python_callable=train_test_split_function,
    dag=dag,
)



DATA_DIR = Path(__file__).parent.parent/'working_data'/'TRAIN'/'train.csv'
Logistic_Regression_task = PythonOperator(
    task_id='Logistic_Regression_task',
    python_callable=Logistic_model,
    op_args=[DATA_DIR],
    dag=dag,
)

best_nn_model_task = PythonOperator(
    task_id='best_nn_model_task',
    python_callable=run_best_model,
    op_args=[DATA_DIR, ''],
    dag=dag,
)
'''
random_forest_model_task = PythonOperator(
    task_id='random_forest_model_task',
    python_callable=random_forest_pipeline,
    op_args=[DATA_DIR],
    dag=dag,
)
'''



# Set task dependencies

data_download_task >> lab_vital_generator_task >> pivoting_task >> [fetch_grouped_sorted_labs_nopneumonia, fetch_grouped_sorted_labs_pneumonia]
fetch_grouped_sorted_labs_pneumonia >> fetch_filled_labs_pneumonia >> fetch_merged_vitals_labs_pneumonia >> fetch_closeby_vitalslabs_pneumonia >> dropduplicates_pneumonia >> load_pneumonia_data_task
load_pneumonia_data_task >> cleaning_task_pneumonia >> label_creation_task_pneumonia >> age_preprocessing_task_pneumonia




fetch_grouped_sorted_labs_nopneumonia >> fetch_filled_labs_nopneumonia >> fetch_merged_vitals_labs_nopneumonia >> fetch_closeby_vitalslabs_nopneumonia 
fetch_closeby_vitalslabs_nopneumonia >> dropduplicates_nopneumonia >> load_nopneumonia_data_task >> cleaning_task_nopneumonia >> label_creation_task_nopneumonia
label_creation_task_nopneumonia >> age_preprocessing_task_nopneumonia



[age_preprocessing_task_pneumonia, age_preprocessing_task_nopneumonia] >> train_test_split_task


train_test_split_task >> [Logistic_Regression_task, best_nn_model_task]



'''
tfx_schema_task
random_forest_model_task
'''

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()