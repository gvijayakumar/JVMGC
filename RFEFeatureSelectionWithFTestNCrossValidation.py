from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from scipy import stats
import numpy as np
import pandas as pd

# Sample Data (Replace with actual data)
np.random.seed(42)
X = pd.DataFrame(np.random.rand(100, 10), columns=[f'Feature_{i}' for i in range(10)])
y = pd.DataFrame(np.random.rand(100, 1), columns=['Target'])

# Split into train/test sets (replace with actual splits)
train_size = int(0.8 * len(X))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Model Initialization
model = LinearRegression()

# Recursive Feature Elimination (RFE)
num_features_to_select = 5
rfe = RFE(model, n_features_to_select=num_features_to_select)
X_train_rfe = rfe.fit_transform(X_train, y_train.values.ravel())
X_test_rfe = rfe.transform(X_test)

# Fit the model on selected features
model.fit(X_train_rfe, y_train)

# Make predictions
y_pred = model.predict(X_test_rfe)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.4f}")

# Cross-Validation (5-Fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_rfe, y_train, cv=kf, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()  # Convert negative MSE to positive
print(f"Cross-Validation MSE: {cv_mse:.4f}")

# F-test for model significance
n, p = X_train_rfe.shape  # n = samples, p = selected features
rss = np.sum((y_train - model.predict(X_train_rfe)) ** 2)  # Residual Sum of Squares
tss = np.sum((y_train - np.mean(y_train)) ** 2)  # Total Sum of Squares
r_squared = 1 - (rss / tss)
f_statistic = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
p_value = 1 - stats.f.cdf(f_statistic, p, n - p - 1)

print(f"F-statistic: {f_statistic:.4f}, p-value: {p_value:.4f}")

# Feature Selection Results
selected_features = X.columns[rfe.support_]
print("Selected Features:", list(selected_features))

# Interpretation
if p_value < 0.05:
    print("The model is statistically significant (p < 0.05).")
else:
    print("The model is NOT statistically significant (p >= 0.05).")