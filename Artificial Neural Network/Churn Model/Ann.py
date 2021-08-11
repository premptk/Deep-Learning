# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 21:15:13 2021

@author: Prem Ranjan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13]
y = dataset.iloc[:,13]

#create dummy variables
geography = pd.get_dummies(X['Geography'],drop_first=True)
gender = pd.get_dummies(X['Gender'],drop_first=True)

#concate the dummy variable to the dataset
X = pd.concat([X, geography, gender], axis=1)
#droping the categorical data
X.drop(['Geography','Gender'], inplace=True, axis=1)

#Spliting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#importing keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

#Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))

#creating second hidden layer 
classifier.add(Dense(units=6, kernel_initializer = 'he_uniform', activation='relu'))

#creating third layer (output layer)
classifier.add(Dense(units=1, kernel_initializer = 'glorot_uniform', activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])

#Fiting the ANN to the training set
model_history = classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 100)

#list all data in history
print(model_history.history.keys())

#summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Evaluating the Model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Calculate the Accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)


