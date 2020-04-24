# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:17:51 2020

@author: 29594
"""

from sklearn.model_selection import train_test_split
from sklearn import datasets
'''载入鸢尾花数据集'''
iris=datasets.load_iris()
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y)

from KNN import KNNClassifier
my_knn_clf=KNNClassifier(k=3)
my_knn_clf.fit(X_train,y_train)
y_predict=my_knn_clf.predict(X_test)

print(sum(y_predict==y_test))
"""预测准确率"""
print(sum(y_predict==y_test)/len(y_test))

