# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 01:35:53 2018

@author: shivam agrawal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset= pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder1=LabelEncoder()
X[:,1]= labelencoder1.fit_transform(X[:,1])

labelencoder2=LabelEncoder()
X[:,2]= labelencoder2.fit_transform(X[:,2])

onehotencoder= OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()

X=X[:,1:]

#preprocessing done


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#split into test and train

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#normalise

from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
#first hidden layer

classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
#second hidden layer

classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#output

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#neural network designed

classifier.fit(X_train,y_train,batch_size=10,epochs=100)

y_pred= classifier.predict(X_test)
y_pred= (y_pred >0.5)

y_pred = np.squeeze(y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


from sklearn.metrics import confusion_matrix,classification_report

matrix=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)
print(matrix)
print(report)














