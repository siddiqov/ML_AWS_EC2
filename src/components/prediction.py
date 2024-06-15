import pickle
import numpy as np

def load_model():
    with open('artifacts/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('artifacts/preprocessor.pkl', 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    return model, preprocessor

def predict_salary(features):
    model, preprocessor = load_model()
    features_transformed = preprocessor.transform([features])
    prediction = model.predict(features_transformed)
    return round(prediction[0], 2)

if __name__ == "__main__":
    sample_features = [32, 'Male', "Bachelor's", 'Software Engineer', 5]
    salary = predict_salary(sample_features)
    print(f"Predicted Salary: ${salary}")
