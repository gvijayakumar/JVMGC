import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Define the data
data = {
    'DefNew (allocated space)(KB)': [576, 576, 576, 576, 576, 0, 576, 576, 576, 576, 576, 576, 576],
    'DefNew (Before GC)(KB)': [512, 576, 576, 575, 576, 0, 576, 575, 575, 575, 575, 575, 575],
    'DefNew (After GC)(KB)': [64, 64, 64, 64, 64, 0, 83, 326, 449, 483, 529, 553, 565],
    'Tenured (allocated space)(KB)': [1408, 1408, 1408, 1408, 1408, 0, 1408, 1408, 1408, 1408, 1408, 1408, 1408],
    'Tenured (Before GC)(KB)': [0, 367, 515, 677, 926, 0, 1177, 1407, 1407, 1407, 1407, 1407, 1407],
    'Tenured (After GC)(KB)': [367, 515, 677, 926, 1177, 0, 1407, 1407, 1407, 1407, 1407, 1407, 1407],
    'GC Pause Time(ms)': [1.466, 0.872, 0.71, 0.903, 0.787, 0.069, 2.877, 3.386, 2.808, 3.327, 2.785, 2.963, 2.576],
    'User(s)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Sys(s)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Real Time(s)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'CPU Time(s)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GC/FULL-GC(1->GC,0->FULL-GC)': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=['GC/FULL-GC(1->GC,0->FULL-GC)'])
y = df['GC/FULL-GC(1->GC,0->FULL-GC)']

# Initialize linear regression model
model = LinearRegression()

# Initialize RFE
rfe = RFE(model, n_features_to_select=3)  # Selecting top 5 features

# Fit RFE
fit = rfe.fit(X, y)

# Display results
print("Selected Features:")
for i in range(len(X.columns)):
    if fit.support_[i]:
        print(X.columns[i])
