#This is the brute force implementation of KNN.
from scipy.spatial.distance import euclidean
import numpy as np
class KNN:
    def __init__(self,k,X_train,Y_train):
        self.k=k
        self.X_train=X_train
        self.Y_train=Y_train
    def get_neighbors(self,point):
        neighbors_idxs=[]
        distArr=[]
        distArr=np.array(distArr)
        for i in range(len(self.X_train)):
            dist=euclidean(point,self.X_train[i])
            distArr=np.append(distArr,dist)
        neighbors_idxs=np.argsort(distArr)[:self.k]
        return neighbors_idxs
    def get_class(self,neighbors_idxs,point):
        pos_n=0
        neg_n=0
        for i in neighbors_idxs:
            if self.Y_train[i]==1:
                pos_n+=1
            else:
                neg_n+=1
        if pos_n>neg_n:
            return 1
        else:
            return -1
    def predict(self,X):
        Y=[]
        Y=np.array(Y)
        for i in range(len(X)):
            point=X[i]
            neighbors=self.get_neighbors(point)
            label=self.get_class(neighbors,point)
            Y=np.append(Y,label)
        return Y
