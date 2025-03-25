# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('..\Sourcecode\Serial_GC\dataset.csv')

# Separate features (X) and target variable (y)
X = df.drop('Throughput(%)', axis=1)
y = df['Throughput(%)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest regressor (as an example model)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Feature Selection with Recursive Feature Elimination (RFE)
rfe = RFE(estimator=regressor, n_features_to_select=2)
X_rfe = rfe.fit_transform(X_train, y_train)
regressor.fit(X_rfe, y_train)
X_test_rfe = rfe.transform(X_test)
y_pred_rfe = regressor.predict(X_test_rfe)
mse_rfe = mean_squared_error(y_test, y_pred_rfe)

# Feature Selection with Principal Component Analysis (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
regressor.fit(X_pca, y_train)
X_test_pca = pca.transform(X_test)
y_pred_pca = regressor.predict(X_test_pca)
mse_pca = mean_squared_error(y_test, y_pred_pca)

# Feature Selection with Univariate Feature Selection (SelectKBest with ANOVA)
kbest = SelectKBest(score_func=f_regression, k=2)
X_kbest = kbest.fit_transform(X_train, y_train)
regressor.fit(X_kbest, y_train)
X_test_kbest = kbest.transform(X_test)
y_pred_kbest = regressor.predict(X_test_kbest)
mse_kbest = mean_squared_error(y_test, y_pred_kbest)

# Print the mean squared errors of different feature selection techniques
print("Mean Squared Error with RFE:", mse_rfe)
print("Mean Squared Error with PCA:", mse_pca)
print("Mean Squared Error with SelectKBest (ANOVA):", mse_kbest)
