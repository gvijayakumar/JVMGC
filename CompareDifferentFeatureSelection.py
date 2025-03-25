import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, mean_absolute_error

# Read data from CSV
file_path = r'C:/vijay/research/JVMGC/Sourcecode/Serial_GC/dataset/Xms2m_Xmx100m_SGC.csv'
df = pd.read_csv(file_path)

# Separate features and target
X = df.drop(columns=['GC/FULL-GC(1->GC,0->FULL-GC)'])
y = df['GC/FULL-GC(1->GC,0->FULL-GC)']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Feature selection techniques
selectors = {
    'RFE': RFE(log_reg, n_features_to_select=5),
    'SelectKBest': SelectKBest(score_func=f_classif, k=5)
}

# Feature selection comparison
for name, selector in selectors.items():
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()]
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Train model with selected features
    log_reg.fit(X_train_selected, y_train)
    y_pred = log_reg.predict(X_test_selected)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Results for {name}:")
    print(f"Selected Features: {selected_features}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print()
