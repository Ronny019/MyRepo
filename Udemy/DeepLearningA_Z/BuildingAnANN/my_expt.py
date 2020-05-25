import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 25, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test,y_pred)
print(cm)
print(acc)

## Evaluating the ANN
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#from keras.models import Sequential
#from keras.layers import Dense
#def build_classifier():
#    classifier = Sequential()
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return classifier
#classifier = KerasClassifier(build_fn = build_classifier, batch_size = 25, epochs = 100)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#print(accuracies)
#mean = accuracies.mean()
#print(mean)
#variance = accuracies.std()
#print(variance)


#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#def build_classifier(optimizer):
#    classifier = Sequential()
#    classifier.add(Dense(units = 120, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#    classifier.add(Dense(units = 60, kernel_initializer = 'uniform', activation = 'relu'))
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return classifier
#classifier = KerasClassifier(build_fn = build_classifier)
#parameters = {'batch_size': [25],
#              'epochs': [100],
#              'optimizer': ['rmsprop']}
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 2)
#grid_search = grid_search.fit(X_train, y_train)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_

#print(best_parameters)
#print(best_accuracy)