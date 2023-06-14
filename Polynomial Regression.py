import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Years.csv')
X = dataset[['Years']]
y = dataset[['Salary']]
dataset.head()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)

plt.scatter(X, y)
plt.plot(X, reg.predict(X), color = 'red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
feature = PolynomialFeatures(degree=4)
poly = feature.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(poly, y)

plt.scatter(X, y)
plt.plot(X, lin_reg.predict(poly), color = 'red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

lin_reg.predict(poly)

