# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:32:12 2024

@author: Admin
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('C:/vijay/research/JVMGC/Sourcecode/Serial_GC/dataset/Xms2m_Xmx17700m_SGC.csv')  # Replace 'your_data_file.csv' with the actual file name

# Split data into features and target variable
X = data.drop(columns=['GC Pause Time(ms)'])  # Features (all columns except 'GC Pause Time(ms)')
y = data['GC Pause Time(ms)']  # Target variable ('GC Pause Time(ms)')

# Feature selection using SelectKBest with f_regression scoring
num_features_to_select = 3  # Adjust this number based on your preference
selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print selected feature names
print("Selected Features:")
for feature in selected_features:
    print(feature)
