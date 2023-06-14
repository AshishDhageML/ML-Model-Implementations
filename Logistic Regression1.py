import pandas as pd
df = pd.read_csv("job.csv")
df.head(10)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['Years']],df.result,train_size=0.8,random_state=10)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


model.fit(X_train, y_train)


y_predicted = model.predict(X_test)
y_predicted

model.score(X_test,y_test)

