import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from data_ingestion import load_data
from data_transformation import transform_data

def evaluate_model(data_path):
    df = load_data(data_path)
    features, target, preprocessor = transform_data(df)
    
    with open('artifacts/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    predictions = model.predict(features)
    mse = mean_squared_error(target, predictions)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    data_path = 'artifacts/data/Salary_Data.csv'
    evaluate_model(data_path)
