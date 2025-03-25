import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Define the folder path
folder_path = r'C:\vijay\research\JVMGC\Sourcecode\Serial_GC\dataset'

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Iterate through each CSV file
for file in csv_files:
    file_path = os.path.join(folder_path, file)

    # Read the data from the CSV file
    df = pd.read_csv(file_path)

    # Separate features (X) and target variable (y)
    # Assuming the first column is the target variable and the rest are features
    target_column = df.columns[0]
    features = df.drop(columns=[target_column])
    target = df[target_column]

    # Initialize a linear regression model
    model = LinearRegression()

    # Initialize RFE with the linear regression model
    rfe = RFE(model, n_features_to_select=1)

    # Fit RFE and get the ranking of each feature
    rfe.fit(features, target)

    # Get the selected features
    selected_features = features.columns[rfe.support_]

    # Print the selected features for the current CSV file
    #print(f"\nFile: {file}")
    print(f"Selected Features: {', '.join(selected_features)}")
