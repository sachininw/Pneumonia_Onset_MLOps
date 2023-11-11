import gdown
import zipfile
from pathlib import Path
 
def download_file_from_google_drive(id, destination):
    url = f'https://drive.google.com/uc?id={id}'
    gdown.download(url, destination, quiet=False)
 
#file_id = '1AOxj32OiSqqiuw0BicR3DGpeQ40qNfsh'
 
def unzip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)


def download():
    file_id = '17I6Z4o-AxHnol02pQgnDvu_pfU54-3pu'
    download_file_from_google_drive(file_id, './data/raw_data.zip')

    zip_file_path = './data/raw_data.zip'
    extract_to_path = './data'

    unzip_file(zip_file_path, extract_to_path)