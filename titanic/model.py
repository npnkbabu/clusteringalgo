# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class model:
    
    def __init__(self,train,test):
        print('model instantiated')
        self.__train ,self.__test = train,test
        self.__X = np.array((self.__train).astype(float))
        self.__tunemodelwithwcss()
        self.__checkpef()
        
    def __createmodel(self,kvalue):
        self.__kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
                                n_clusters=kvalue, n_init=10, n_jobs=1, precompute_distances='auto',
                                random_state=None, tol=0.0001, verbose=0)
        self.__kmeans.fit(self.__X)
        
    def __tunemodelwithwcss(self):
        print('started model tuning')
        wcss=[]
        for i in range(1,11):
            k = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
                       n_clusters=i, n_init=10, n_jobs=1, precompute_distances='auto',
                       random_state=None, tol=0.0001, verbose=0)
            k.fit(self.__X)
            wcss.append(k.inertia_)
        plt.plot(range(1,11),wcss)
        plt.xlabel('Sum of sq distances from centroid')
        plt.ylabel('no. of centroids')
        plt.title('Within sum of squares WCSS elbow method')
        plt.show()
        self.__createmodel(2)
    def __checkpef(self):
        y = np.array(self.__train['Survived'])
        correct = 0
        for i in range(len(self.__X)):
            pred_me = np.array(self.__X[i].astype(float))
            pred_me = pred_me.reshape(-1,len(pred_me))
            pred = self.__kmeans.predict(pred_me)
            if(pred == y[i]):
                correct += 1
        print(correct/len(self.__X))
        

            

