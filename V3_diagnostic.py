# V3_diagnostic.py

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to diagnose data issues

def diagnose_data_issues(data):
    # Check for missing values
    missing_values = data.isnull().sum()
    print("Missing values at each column:\n", missing_values)

    # Check for data types
    data_types = data.dtypes
    print("Data types of each column:\n", data_types)

# Function to diagnose model learning issues

def diagnose_model_issues(y_true, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

# Example usage (This part can be modified as per use case)

if __name__ == '__main__':
    # Load your dataset here
    # data = pd.read_csv('your_data.csv')
    # diagnose_data_issues(data)
    
    # Load your model predictions (y_true, y_pred)
    # y_true = [0, 1, 0, 1]
    # y_pred = [0, 0, 1, 1]
    # diagnose_model_issues(y_true, y_pred)