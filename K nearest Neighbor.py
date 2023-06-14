
import pandas as pd


dataset = pd.read_csv('Car.csv')
X = dataset[['Age','Income']]
y = dataset[['Car']]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

X_train

X_test

y_train

y_test

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(X_train, y_train)

print(classifier.predict([[60,8500]]))

classifier.score(X_test,y_test)







