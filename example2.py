#import pandas
#import matplotlib.pyplot as plt
#dataset = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#plt.plot(dataset)
#plt.show()

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# fix random seed for reproducibility
numpy.random.seed(7)

head = ["id","Time", "CBFV.C1", "ABP", "CBFV.C2", "End-Tindal", "CrCP.C1", "RAP.C1", "Heart Rate", "Sistolic Pressure",
        "Diastolic Pressure", "POURCELOT'S INDEX  CHANNEL 1", "MEAN SQUARE ERROR FOR P-V CH. 1", "CrCP.C2", "RAP.C2",
        "SYSTOLIC  VEL. CH.1", "DIASTOLIC VEL. CH.1", "SYSTOLIC VELOCITY CH. 2", "DIASTOLIC VELOCITY CH.2",
        "POURCELOT INDEX CH. 2", "MSE FOR P-V  CH. 2"]

# load the dataset
dataframe = pandas.read_csv('HC092101.PAR', names = head)
dataset1 = dataframe["ABP"].values
dataset1 = dataset1.astype('float32')

dataset2 = dataframe["CBFV.C1"].values
dataset2 = dataset1.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(dataset1)
dataset2 = scaler.fit_transform(dataset2)

# split into train and test sets
train_size = int(len(dataset1) * 0.67)
test_size = len(dataset1) - train_size
train1, test1 = dataset1[0:train_size,:], dataset1[train_size:len(dataset1),:]
train2, test2 = dataset2[0:train_size,:], dataset2[train_size:len(dataset2),:]
print(len(train1), len(test2))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 2
trainX1, trainY1 = create_dataset(train1, look_back)
testX1, testY1 = create_dataset(test1, look_back)
trainX2, trainY2 = create_dataset(train2, look_back)
testX2, testY2 = create_dataset(test2, look_back)

trainX1 = np.reshape(trainX1, (trainX1.shape[0], 1, trainX1.shape[1]))
testX1 = np.reshape(testX1, (testX1.shape[0], 1, testX1.shape[1]))
trainX2 = np.reshape(trainX2, (trainX2.shape[0], 1, trainX2.shape[1]))
testX2 = np.reshape(testX2, (testX2.shape[0], 1, testX2.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#print(trainX.shape)
#print(trainY.shape)
model.fit(trainX1, trainY2, epochs=100, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX1)
testPredict = model.predict(testX1)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY2])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY2])
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset2)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset2)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset2)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset2))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

#tienes que predecir el cbfv.ch1 en base al abp
#abp es la presión arterial media y el cbfv.ch1 es la valocidad de flujo sanguíneo cerebral en el canala 1
#tienes que predecir el cbfv.ch1 en base al abp

