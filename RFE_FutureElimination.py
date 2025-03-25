# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Boston Housing dataset as an example
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=["target"])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Use RFE for feature extraction
# In this example, I'm selecting the top 5 features
num_features_to_select = 5
rfe = RFE(model, n_features_to_select=num_features_to_select)
X_train_rfe = rfe.fit_transform(X_train, y_train.values.ravel())
X_test_rfe = rfe.transform(X_test)

# Fit the model on the selected features
model.fit(X_train_rfe, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_rfe)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on the test set: {mse}")

# Get the selected features
selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)
