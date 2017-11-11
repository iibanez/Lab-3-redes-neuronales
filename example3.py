import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation

numpy.random.seed(7)

# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility

def remuestre(cantidad, columna, dataframe):
	suma = 0
	contador = 0
	eliminar = []
	for i in range(len(dataframe)):
		contador += 1
		suma += dataframe[columna][i]
		if (contador == cantidad):
			dataframe[columna][i] = suma / cantidad
			contador = 0
			suma = 0
		else:
			eliminar.append(i)
	dataframe = dataframe.drop(dataframe.index[eliminar])
	return dataframe

# load the dataset

#dataframe = pandas.read_csv('HC092101.PAR', names = head)
cantidad = 3

dataframe = pandas.read_csv('HC092101.PAR', usecols=[3], engine='python', skipfooter=3, header=None)
dataframe = remuestre(cantidad,3, dataframe)
dataset1 = dataframe.values
dataset1 = dataset1.astype('float32')

dataframe = pandas.read_csv('HC092101.PAR', usecols=[2], engine='python', skipfooter=3, header=None)
dataframe = remuestre(cantidad, 2,dataframe)
dataset2 = dataframe.values
dataset2 = dataset2.astype('float32')

# normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(dataset1)
dataset2 = scaler.fit_transform(dataset2)

# split into train and test sets
#train_size = int(len(dataset1) * 0.67)
#test_size = len(dataset1) - train_size
#train1, test1 = dataset1[0:train_size,:], dataset1[train_size:len(dataset1),:]
#train2, test2 = dataset2[0:train_size,:], dataset2[train_size:len(dataset2),:]
#print(len(train1), len(test2))

# reshape into X=t and Y=t+1

look_back = 8
#trainX1, trainY1 = create_dataset(train1, look_back) #TrainX1 es la presion Entrenamiento
#testX1, testY1 = create_dataset(test1, look_back) #TestX1 es la presion de test
#trainX2, trainY2 = create_dataset(train2, look_back) #TrainX2 autorregulacion de entrenamiento
#testX2, testY2 = create_dataset(test2, look_back)#testY2 autorregulacion de test

abpX, abpY = create_dataset(dataset1, look_back)
cbfvX, cbfvY = create_dataset(dataset2, look_back)
# reshape input to be [samples, time steps, feature


#trainX1 = numpy.reshape(trainX1, (trainX1.shape[0], 1, trainX1.shape[1]))
#testX1 = numpy.reshape(testX1, (testX1.shape[0], 1, testX1.shape[1]))
abpX = numpy.reshape(abpX, (abpX.shape[0], 1, abpX.shape[1]))
cbfvX = numpy.reshape(cbfvX, (cbfvX.shape[0], 1, cbfvX.shape[1]))

cv = cross_validation.KFold(len(abpY), n_folds=3)

# se inicia la validacion cruzada
for traincv, testcv in cv:
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(5, input_shape=(1, look_back)))
	model.add(Dense(1, activation='relu'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(abpX[traincv], cbfvY[traincv], epochs=20, batch_size=1, verbose=2)
	trainPredict = model.predict([abpX[traincv]])
	testPredict = model.predict([abpX[testcv]])
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([cbfvY[traincv]])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([cbfvY[testcv]])
	corr = numpy.corrcoef(cbfvY[testcv], testPredict[:,0])
	print('Test Score: %.2f CORRELACION' % (corr[1][0]))

# shift train predictions for plotting

trainPredictPlot = numpy.empty_like(dataset2)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting

testPredictPlot = numpy.empty_like(dataset2)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)-1:len(dataset2)-1, :] = testPredict

# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset2))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()