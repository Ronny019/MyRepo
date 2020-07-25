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
look_back= 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Lets create a LSTM(RNN) model
model = Sequential()
model.add(LSTM(4, input_shape = (1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# fit the model
model.fit(trainX, trainY, batch_size =1, verbose = 2,epochs = 100)

#make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Reverse the predicted value to actual values

trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

trainY = scaler.inverse_transform([trainY])
testY = scaler.inverse_transform([testY])

# Calculate RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train : %.2f RMSE' % (trainScore))

testScore= math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test : %.2f RMSE' % (testScore))