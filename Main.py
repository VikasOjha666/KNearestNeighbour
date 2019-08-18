import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN_implementation import KNN

df=pd.read_csv('iris.csv',header=None)
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
X=df.iloc[0:100,[0,2]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_std_train=sc.transform(X_train)
X_std_test=sc.transform(X_test)

X_train=np.array(X_std_train)
Y_train=np.array(y_train)

knn=KNN(30,X_train,Y_train)
Y=knn.predict(X_std_test)

from sklearn.metrics import accuracy_score
print("Accuracy=",accuracy_score(y_test,Y))

X_combined_std = np.vstack((X_std_train, X_std_test))
y_combined = np.hstack((Y_train, y_test))

def plot_decision_regions(X, y, classifier, resolution=0.02):
#This functions has been taken from python Machine Learning book by Sebestian Rashchka.
# setup marker generator and color map
 markers = ('s', 'x', 'o', '^', 'v')
 colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
 cmap = ListedColormap(colors[:len(np.unique(y))])
 # plot the decision surface
 x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
 x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
 xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
 np.arange(x2_min, x2_max, resolution))
 Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
 Z = Z.reshape(xx1.shape)
 plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
 plt.xlim(xx1.min(), xx1.max())
 plt.ylim(xx2.min(), xx2.max())

 # plot class samples
 for idx, cl in enumerate(np.unique(y)):
  plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)




plot_decision_regions(X_combined_std, y_combined, classifier=knn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
