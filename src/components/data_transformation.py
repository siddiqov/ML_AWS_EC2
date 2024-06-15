import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def transform_data(df):
    # Drop rows where the target variable 'Salary' is NaN
    df = df.dropna()
    features = df.drop('Salary', axis=1)
    target = df['Salary']
    
    numeric_features = ['Age', 'Years of Experience']
    categorical_features = ['Gender', 'Education Level', 'Job Title']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
        ('onehot', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    features_transformed = preprocessor.fit_transform(features)
    
    return features_transformed, target, preprocessor

if __name__ == "__main__":
    data_path = 'artifacts/data/Salary_Data.csv'
    df = pd.read_csv(data_path)
    features, target, preprocessor = transform_data(df)
    print(features[:5])
