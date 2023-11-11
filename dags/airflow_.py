# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.preprocessing import age_preprocessing, load_data
from src.labs_vitals_generator import read_concatenate_save
from src.pivoting import pivot_lab_results_and_save
from src.combine_lab_vitals import fetch_grouped_sorted_labs, fetch_filled_labs, fetch_merged_vitals_labs, fetch_closeby_vitalslabs, dropduplicates
from src.data_download import download
from src.cleaning import apply_conversion
from src.label_creation import label_creation_apply

import os

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



# Set task dependencies

data_download_task >> lab_vital_generator_task >> pivoting_task >> [fetch_grouped_sorted_labs_nopneumonia, fetch_grouped_sorted_labs_pneumonia]
fetch_grouped_sorted_labs_pneumonia >> fetch_filled_labs_pneumonia >> fetch_merged_vitals_labs_pneumonia >> fetch_closeby_vitalslabs_pneumonia >> dropduplicates_pneumonia >> load_pneumonia_data_task
load_pneumonia_data_task >> cleaning_task_pneumonia >> label_creation_task_pneumonia




fetch_grouped_sorted_labs_nopneumonia >> fetch_filled_labs_nopneumonia >> fetch_merged_vitals_labs_nopneumonia >> fetch_closeby_vitalslabs_nopneumonia 
fetch_closeby_vitalslabs_nopneumonia >> dropduplicates_nopneumonia >> load_nopneumonia_data_task >> cleaning_task_nopneumonia >> label_creation_task_nopneumonia


# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()