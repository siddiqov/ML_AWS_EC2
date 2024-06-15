import os
import shutil
import pandas as pd

def ingest_data(source_path, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    shutil.copy(source_path, dest_path)
    print(f"Data ingested successfully from {source_path} to {dest_path}")

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

if __name__ == "__main__":
    source_file = r'D:\ML_DS_DEVOP\ML_AWS_EC2\ML_AWS_EC2\Salary_Data.csv'
    dest_folder = r'artifacts/data'
    dest_file = os.path.join(dest_folder, 'Salary_Data.csv')
    ingest_data(source_file, dest_folder)

    # Load and print the data
    data = load_data(dest_file)
    print(data.head())
