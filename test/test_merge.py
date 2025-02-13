import unittest
import pandas as pd
import os
from pathlib import Path
import pickle

from dags.src import cleaning
from dags.src import combine_lab_vitals
from dags.src import data_download
import pytest


@pytest.fixture
def test_convert_to_float_and_merge():
    # Test the conversion function
    input_list = "[1, 1,1,1,1,1]"
    result = cleaning.convert_to_float_and_merge(input_list)
    assert result == 1

    input_not_list = "abc"
    result_not_list = cleaning.convert_to_float_and_merge(input_not_list)
    assert result_not_list == "abc"
    
    
    item = None
    result = cleaning.convert_to_float_and_merge(item)
    assert result is None
    
    
@pytest.fixture
def test_drop_duplicates():
    # Create a sample DataFrame with duplicates
    df = pd.DataFrame({'A': [1, 2, 2, 3, 4], 'B': [5, 6, 6, 7, 8]})

    # Call the dropduplicates function
    result = combine_lab_vitals.dropduplicates('pneumonia', df)

    # Check that duplicates are dropped
    assert len(result) == 4
    assert result.duplicated().sum() == 0

@pytest.fixture
# Downloads file from Google Drive with valid ID and destination
def test_valid_id_and_destination():
    file_id = '17I6Z4o-AxHnol02pQgnDvu_pfU54-3pu'
    destination = './data/raw_data.zip'
    data_download.download_file_from_google_drive(file_id, destination)
    assert os.path.exists(destination)