
import pandas as pd



dataset = pd.read_csv('age1.csv')
X = dataset[['Age']]
Y = dataset[['Income']]
dataset



from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, Y)



regressor.predict([[70]])



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)



from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)



reg.predict(X_test)

