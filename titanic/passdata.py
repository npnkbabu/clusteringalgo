# -*- coding: utf-8 -*-
'''
cluster data into 2 groups survived and not-survived and we can validate
with the ouptut data. drop existing survivied column
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class passdata:
    __trainfile = 'titanic/data/train.csv'
    __testfile = 'titanic/data/test.csv'
    __unncols = ['Name','Ticket','Cabin','Embarked','PassengerId']
    __histCols = ['Age','Fare']
    
    def __init__(self):
        print('passdata instantiated')
        self.__train = pd.read_csv(self.__trainfile)
        self.__test = pd.read_csv(self.__testfile)
        
        
    def preprocess(self):
        print('----variable identification----')
        print('dtypes of columns ',self.__train.dtypes)
        print('sample data')
        #print(self.__train.head(1))
        print('drop unneccesary column name')
        self.__train.drop(self.__unncols,axis=1,inplace=True)
        self.__test.drop(self.__unncols,axis=1,inplace=True)
        
        print('handle missing values with mean')
        self.__train.fillna(self.__train.mean(),inplace=True)
        self.__test.fillna(self.__test.mean(),inplace=True)
        print('train nulls : \n' ,self.__train.isna().sum())
        print('test nulls : \n',self.__test.isna().sum())
        
        print('----univariate analysis----')
        print('Frequency distribution, central tendency and dispersion')
        print('frequency distribution : historgram and box plot for numerical vaiables')
        
        for col in self.__histCols:
            print('hist plot for ',col)
            #sns.distplot(self.__train[col].values)
            plt.hist(self.__train[col])
            plt.xlabel(col)
            plt.title('hist plot for '+col)
            plt.show()
        
        print('bivariate analysis with survived count')
        print('sex vs survived')
        plt.subplot(1,2,1)
        plt.hist(self.__train[self.__train['Survived']==0]['Age'])
        plt.title('Not Survived')
        plt.xlim(0,101)
        plt.xticks(np.arange(0,101,20))
        plt.yticks(np.arange(0,301,20))
        plt.subplot(1,2,2)
        plt.hist(self.__train[self.__train['Survived']==1]['Age'])
        plt.title('Survived')
        plt.xlim(0,101)
        plt.yticks(np.arange(0,151,20))
        plt.xticks(np.arange(0,101,20))
        plt.show()
        
        print('scatter plot age vs fare for survived and not survived')
        survived = self.__train[self.__train['Survived']==1]
        notsurvived = self.__train[self.__train['Survived']==0]
        plt.scatter(np.array(survived['Fare']),np.array(survived['Age']),color='green',label='Survived')
        plt.scatter(np.array(notsurvived['Fare']),np.array(notsurvived['Age']),color='red',label='Not-Survived')
        plt.xticks(np.arange(0,101,20))
        plt.xlim(1,101)
        plt.legend()
        plt.show()
        print('plot shows clusters are not sperical and they are not equal size also so we can not use k-means')
        
        #variable creation on categorical variables.
        self.__lblencod = LabelEncoder()
        self.__train['Sex']=self.__lblencod.fit_transform(self.__train['Sex'])
        self.__test['Sex']=self.__lblencod.fit_transform(self.__test['Sex'])
        #print(self.__train.head(2))
        print('using minmaxscalar for Age and Fare')
        self.__minmaxscal = MinMaxScaler()
        self.__train['Age'] = self.__minmaxscal.fit_transform(self.__train[['Age']])
        self.__train['Fare'] = self.__minmaxscal.fit_transform(self.__train[['Fare']])
        self.__test['Age'] = self.__minmaxscal.fit_transform(self.__test[['Age']])
        self.__test['Fare'] = self.__minmaxscal.fit_transform(self.__test[['Fare']])
        '''
        self.__train['Fare']= self.__minmaxscal.fit_transform(self.__train['Fare'].values)
        self.__test['Age']= self.__minmaxscal.fit_transform(self.__test['Age'].values)
        self.__test['Fare']= self.__minmaxscal.fit_transform(self.__test['Fare'].values)
        '''
        print('preprocessing completed')
        return self.__train,self.__test



