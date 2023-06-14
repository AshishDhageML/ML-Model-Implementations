import pandas as pd

dataset = pd.read_csv('Bike.csv')
X = dataset[['Age','Income']]
y = dataset[['Bike']]
dataset.head(10)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

classifier.predict(X_test)

classifier.score(X_test,y_test)

