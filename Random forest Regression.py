import pandas as pd

dataset = pd.read_csv('age1.csv')
X = dataset[['Age']]
Y = dataset[['Income']]
dataset

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X, Y)

regressor.predict([[70]])

regressor.predict(X)

regressor.score(X,Y)

