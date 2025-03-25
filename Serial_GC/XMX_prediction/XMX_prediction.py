from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import SGDRegressor


x_a=[]
y_a=[]
data = pd.read_csv("../dataset.csv")
X = data.iloc[:,:-4]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=.40)
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


regressor = HuberRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Huber : "+str(r2_score(y_test, y_pred)*100))
y_a.append(r2_score(y_test, y_pred)*100)
x_a.append("Huber")

regressor = TweedieRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Tweedie : "+str(r2_score(y_test, y_pred)*100))
y_a.append(r2_score(y_test, y_pred)*100)
x_a.append("Tweedie")


regressor = SGDRegressor(random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("SGD: "+str(r2_score(y_test, y_pred)*100))
y_a.append(r2_score(y_test, y_pred)*100)
x_a.append("SGD")


'''
print("DecisionTreeRegressor : "+str(r2_score(y_test, y_pred)*100))
y_a.append(r2_score(y_test, y_pred)*100)
x_a.append("DecisionTree")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
rfr = RandomForestRegressor(n_estimators = 100)
rfr.fit(X_train,y_train)
y_pred = rfr.predict(X_test)
s = mean_squared_log_error(y_test, y_pred)
accuracy = 1 - s
print("RandomForestRegressor : "+str(accuracy*100))
y_a.append(accuracy*100)
x_a.append("RandomForest")
classifier = LogisticRegression(random_state = 10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("LogisticRegression : "+str(r2_score(y_test, y_pred)*100))
y_a.append(r2_score(y_test, y_pred)*100)
x_a.append("Logistic")
lin_P = LinearRegression()

lin_P.fit(X_train, y_train)
y_pred_P = lin_P.predict(X_test)
print("LinearRegression : "+str(r2_score(y_test, y_pred_P)*100))
y_a.append(r2_score(y_test, y_pred_P)*100)
x_a.append("Linear")
'''
plt.bar(x_a, y_a, width=0.25, color=['red'])
plt.xlabel('Algorithms')
plt.ylabel('Accuracy(0%-100%)')
plt.title('Maximum Heap Size(Xmx) Prediction Graph')
plt.legend()
plt.show()
