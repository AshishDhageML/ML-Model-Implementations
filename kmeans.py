from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df = pd.read_csv("Salary.csv")
df.head()

plt.scatter(df.Age,df['Salary'])
plt.xlabel('Age')
plt.ylabel('Salary')

km = KMeans(n_clusters=3)
predicted = km.fit_predict(df[['Age','Salary']])
predicted

df['cluster']=predicted
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Salary'],color='green')
plt.scatter(df2.Age,df2['Salary'],color='red')
plt.scatter(df3.Age,df3['Salary'],color='blue')
plt.xlabel('Age')
plt.ylabel('Salary')

scale = MinMaxScaler()

scale.fit(df[['Salary']])
df['Salary'] = scale.transform(df[['Salary']])

scale.fit(df[['Age']])
df['Age'] = scale.transform(df[['Age']])

km = KMeans(n_clusters=3)
predicted = km.fit_predict(df[['Age','Salary']])
predicted

df = df.drop(['cluster'], axis='columns')

df['cluster']=predicted
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Salary'],color='green')
plt.scatter(df2.Age,df2['Salary'],color='red')
plt.scatter(df3.Age,df3['Salary'],color='blue')
plt.xlabel('Age')
plt.ylabel('Salary')

km.cluster_centers_

