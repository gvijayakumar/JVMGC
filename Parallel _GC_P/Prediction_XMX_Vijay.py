from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_log_error

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import SGDRegressor

x_a_P=[]
y_a_P=[]
data_P = pd.read_csv("dataset_P1.csv")
X_P = data_P.iloc[:,:-4]
y_P = data_P.iloc[:, -1]
X_train_P, X_test_P, y_train_P, y_test_P = train_test_split(X_P, y_P,random_state=42, test_size=.20)
'''
classifier_P = LogisticRegression(random_state = 10)
classifier_P.fit(X_train_P, y_train_P)
y_pred_P = classifier_P.predict(X_test_P)
print("LogisticRegression : "+str(r2_score(y_test_P, y_pred_P)*100))
y_a_P.append(r2_score(y_test_P, y_pred_P)*100)
x_a_P.append("Logistic")
'''


regressor_P = HuberRegressor()
regressor_P.fit(X_train_P, y_train_P)
y_pred_P = regressor_P.predict(X_test_P)
print("Huber : "+str(r2_score(y_test_P, y_pred_P)*100))
y_a_P.append(r2_score(y_test_P, y_pred_P)*100)
x_a_P.append("Huber")

regressor_P = TweedieRegressor()
regressor_P.fit(X_train_P, y_train_P)
y_pred_P = regressor_P.predict(X_test_P)
print("TweedieRegressor : "+str(r2_score(y_test_P, y_pred_P)*100))
y_a_P.append(r2_score(y_test_P, y_pred_P)*100)
x_a_P.append("TweedieRegressor")

regressor_P = SGDRegressor()
regressor_P.fit(X_train_P, y_train_P)
y_pred_P = regressor_P.predict(X_test_P)
print("SGDRegressor : "+str(r2_score(y_test_P, y_pred_P)*100))
y_a_P.append(r2_score(y_test_P, y_pred_P)*100)
x_a_P.append("SGDRegressor")

'''
sc_P = StandardScaler()
X_train_P = sc_P.fit_transform(X_train_P)
X_test_P = sc_P.transform(X_test_P)
rfr_P = RandomForestRegressor(n_estimators = 100)
rfr_P.fit(X_train_P,y_train_P)
y_pred_P = rfr_P.predict(X_test_P)
s_P = mean_squared_log_error(y_test_P, y_pred_P)
accuracy_P = 1 - s_P
print("RandomForestRegressor : "+str(accuracy_P*100))
y_a_P.append(accuracy_P*100)
x_a_P.append("RandomForest")
'''
plt.bar(x_a_P, y_a_P, width=0.25, color=['xkcd:sky blue'])
plt.xlabel('Algorithms')
plt.ylabel('Accuracy(0%-100%)')
plt.title('Maximum Heap Size(Xmx) Prediction Graph')
#plt.legend()
plt.show()

