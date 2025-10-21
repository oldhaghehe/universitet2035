

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE # Will need to install imblearn

def load_data(file_path):
    """
    Loads the dataset from the specified file path.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def handle_implicit_missing(df):
    """
    Handles implicit missing values by replacing specific strings with 'No'.
    This function is generalized but based on the problem description
    (e.g., 'No internet service' -> 'No').
    """
    df_copy = df.copy()
    # For this dataset, 'International plan' and 'Voice mail plan' already have 'No'/'Yes'
    # No specific implicit missing values found in initial EDA that need replacement here.
    return df_copy

def get_preprocessor_pipeline(numerical_features, categorical_features):
    """
    Creates and returns a scikit-learn preprocessing pipeline using ColumnTransformer.
    """
    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any) - not expected here
    )

    # Create the full preprocessing pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    return pipeline

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Orchestrates the full data preprocessing, including splitting, scaling, encoding, and SMOTE.
    Returns processed training and testing data.
    """
    # Convert 'Churn' to numerical (0/1)
    df['Churn'] = df['Churn'].astype(int)

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Get the preprocessing pipeline
    preprocessor_pipeline = get_preprocessor_pipeline(numerical_features, categorical_features)

    print("\n--- Applying Preprocessing Pipeline ---")
    X_train_processed = preprocessor_pipeline.fit_transform(X_train)
    X_test_processed = preprocessor_pipeline.transform(X_test)

    print(f"Shape of X_train_processed: {X_train_processed.shape}")
    print(f"Shape of X_test_processed: {X_test_processed.shape}")

    # Apply SMOTE to the training data
    print("\n--- Applying SMOTE to Training Data ---")
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    print(f"Shape of X_train_resampled (after SMOTE): {X_train_resampled.shape}")
    print(f"Shape of y_train_resampled (after SMOTE): {y_train_resampled.shape}")
    print("Churn distribution in y_train_resampled:")
    print(pd.Series(y_train_resampled).value_counts(normalize=True))

    print("\nPreprocessing complete. Data is ready for model training.")

    return X_train_resampled, X_test_processed, y_train_resampled, y_test, preprocessor_pipeline, numerical_features, categorical_features

if __name__ == "__main__":
    file_path = "churn-bigml-80.csv"
    df = load_data(file_path)

    if df is not None:
        df_processed = handle_implicit_missing(df.copy())
        X_train, X_test, y_train, y_test, preprocessor, num_feats, cat_feats = preprocess_data(df_processed)
