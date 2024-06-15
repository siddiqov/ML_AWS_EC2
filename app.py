import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model and preprocessor
model = pickle.load(open('artifacts/model.pkl', 'rb'))
preprocessor = pickle.load(open('artifacts/preprocessor.pkl', 'rb'))

# Load the data and extract unique values
data_path = 'artifacts/data/Salary_Data.csv'
df = pd.read_csv(data_path)
unique_genders = df['Gender'].unique().tolist()
unique_education_levels = df['Education Level'].unique().tolist()
unique_job_titles = df['Job Title'].unique().tolist()

@app.route('/')
def home():
    return render_template('index.html', 
                           genders=unique_genders, 
                           education_levels=unique_education_levels, 
                           job_titles=unique_job_titles)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age = int(request.form['Age'])
    years_of_experience = int(request.form['Years of Experience'])
    gender = request.form['Gender']
    education_level = request.form['Education Level']
    job_title = request.form['Job Title']
    
    input_data = pd.DataFrame([[age, years_of_experience, gender, education_level, job_title]],
                              columns=['Age', 'Years of Experience', 'Gender', 'Education Level', 'Job Title'])
    
    features = preprocessor.transform(input_data)
    prediction = model.predict(features)

    output = round(prediction[0], 2)

    return render_template('index.html', 
                           genders=unique_genders, 
                           education_levels=unique_education_levels, 
                           job_titles=unique_job_titles, 
                           prediction_text=f'Employee Salary should be $ {output}')

if __name__ == "__main__":
    app.run(debug=True)
