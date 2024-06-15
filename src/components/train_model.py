import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from data_ingestion import load_data
from data_transformation import transform_data

def train_model(data_path):
    df = load_data(data_path)
    
    # Drop rows where the target variable 'Salary' is NaN
    #df = df.dropna(subset=['Salary'])
    
    features, target, preprocessor = transform_data(df)
    
    model = LinearRegression()
    model.fit(features, target)
    
    with open('artifacts/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('artifacts/preprocessor.pkl', 'wb') as preprocessor_file:
        pickle.dump(preprocessor, preprocessor_file)
    
    print("Model training completed and saved.")

if __name__ == "__main__":
    data_path = 'artifacts/data/Salary_Data.csv'
    train_model(data_path)
