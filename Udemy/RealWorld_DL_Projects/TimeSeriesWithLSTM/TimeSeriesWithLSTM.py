import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# for reproducability of the results, lets fix a seed function
np.random.seed(1234)

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python',skipfooter=3)

dataset = dataset.values
dataset = dataset.astype('float32')

#Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

#spliting the train and test set
train_size = int(len(dataset)*0.67)
test_size = len(dataset)-train_size

train, test = dataset[0:train_size, :], dataset[train_size: len(dataset), :]

# create dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [],[]
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)


#Reshape dataset X= current time, Y= future time 
look_back=1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))