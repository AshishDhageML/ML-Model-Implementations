
import pandas as pd



dataset = pd.read_csv('Bike.csv')
X = dataset[['Age','Income']]
y = dataset[['Bike']]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)




from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)



classifier.predict(X_test)



classifier.score(X_test,y_test)





