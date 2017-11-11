import numpy
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import argparse

numpy.random.seed(7)

# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):
    """
    Se generan los instantes en tiempos t hacia atras
    para lograr predecir el instante t

    :param dataset: conjunto de datos completos
    :param look_back: cuantos instantes de tiempo t hacia atras seran utilizados
    :return: una matriz con los instantes de tiempo hacia atras y el instante presente
    """

    dataX = []
    dataY = []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def remuestre(cantidad, columna, dataframe):
    """
    Se hace un re muestreo en el dataframe disminuyendo la cantidad total
    de datos
    :param cantidad: cantidad de datos que seran promediados
    :param columna: comuna del dataframe en la que estamos trabajando
    :param dataframe: el dataframe con el que se esta trabajando
    :return: dataframe: es el dataframe ya remuestrado
    """
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


def cross_validation(dataset1, dataset2, fold, activacion, epocas, neuronas, look_back):
    """
    Se aplica valicacion cruzada en los datos para predecir la senal
    :param dataset1: datos con la presion
    :param dataset2: datos con el flujo sanguineo
    :param fold: cantidad de folds
    :param activacion: funcion de activacion utilizada
    :param optimizacion: funcion de optimizacion utilizada
    :param epocas: epocas en LSTM
    :param neuronas: cantidad de neuronas utilizadas
    :param look_back: cantidad de datos del pasado seran  utilizados para predecir el futuro
    """
    if(activacion == 0):
        act = 'sigmoid'
    else:
        act = 'relu'

    optimizacion = 'adam'

    print(
        "Una LSTM con %i neuronas, funcion activacion: %s , metodo optimizacion pesos: %s, numero de iteraciones: %i, numero de Folds: %i" % (
            neuronas, activacion, optimizacion, epocas, fold))

    if(fold > 1):
        abpX, abpY = create_dataset(dataset1, look_back)
        cbfvX, cbfvY = create_dataset(dataset2, look_back)


        abpX = numpy.reshape(abpX, (abpX.shape[0], 1, abpX.shape[1]))

        kf = KFold(n_splits=fold)

        correlacionTotal = 0
        # se inicia la validacion cruzada
        it = 0
        for traincv, testcv in kf.split(abpY):
            # create and fit the LSTM network
            csv_logger = keras.callbacks.CSVLogger("training"+str(it)+".log", separator=',', append=False)
            model = Sequential()
            model.add(LSTM(neuronas, input_shape=(1, look_back)))
            model.add(Dense(1, activation=act))
            model.compile(loss='mean_squared_error', optimizer=optimizacion)
            model.fit(abpX[traincv], cbfvY[traincv], epochs=epocas, batch_size=1, verbose=2, callbacks=[csv_logger])
            trainPredict = model.predict([abpX[traincv]])
            testPredict = model.predict([abpX[testcv]])
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([cbfvY[traincv]])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([cbfvY[testcv]])
            corr = numpy.corrcoef(cbfvY[testcv], testPredict[:, 0])
            correlacionTotal+=corr[1][0]
            it+=1

        print('Test Score: %.2f CORRELACION' % (correlacionTotal/fold))

        for i in range(it):
            registro = pandas.read_csv("training"+str(i)+".log", sep=",")

            if(i == 0):
                suma = registro["loss"].values
            else:
                suma+= registro["loss"].values

        plt.plot(registro["epoch"].values, suma/it)
        plt.title("Error LSTM con %i neuronas, correlacion: %.2f" % (neuronas,correlacionTotal/fold * 100))
        plt.ylabel("Error")
        plt.xlabel("Epocas")
        plt.show()

    else:
        keras.callbacks.Callback()
        train_size = int(len(dataset1) * 0.67)
        test_size = len(dataset1) - train_size
        train1, test1 = dataset1[0:train_size, :], dataset1[train_size:len(dataset1), :]
        train2, test2 = dataset2[0:train_size, :], dataset2[train_size:len(dataset2), :]
        print(len(train1), len(test2))

        trainX1, trainY1 = create_dataset(train1, look_back)  # TrainX1 es la presion Entrenamiento
        testX1, testY1 = create_dataset(test1, look_back)  # TestX1 es la presion de test
        trainX2, trainY2 = create_dataset(train2, look_back)  # TrainX2 autorregulacion de entrenamiento
        testX2, testY2 = create_dataset(test2, look_back)  # testY2 autorregulacion de test

        # reshape input to be [samples, time steps, features]

        trainX1 = numpy.reshape(trainX1, (trainX1.shape[0], 1, trainX1.shape[1]))
        testX1 = numpy.reshape(testX1, (testX1.shape[0], 1, testX1.shape[1]))

        # create and fit the LSTM network
        csv_logger = keras.callbacks.CSVLogger('training.log', separator=',', append=False)
        model = Sequential()
        model.add(LSTM(neuronas, input_shape=(1, look_back)))
        model.add(Dense(1, activation=act))
        model.compile(loss='mean_squared_error', optimizer=optimizacion)
        model.fit(trainX1, trainY2, epochs=epocas, batch_size=1, verbose=2, callbacks=[csv_logger])

        # make predictions

        trainPredict = model.predict(trainX1)
        testPredict = model.predict(testX1)

        # invert predictions

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY2])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY2])

        # calculate root mean squared error
        corr = numpy.corrcoef(testY2, testPredict[:, 0])
        print('Test Score: %.2f CORRELACION' % (corr[1][0]))

        # shift train predictions for plotting

        trainPredictPlot = numpy.empty_like(dataset2)
        trainPredictPlot[:, :] = numpy.nan
        trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

        # shift test predictions for plotting

        testPredictPlot = numpy.empty_like(dataset2)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[len(trainPredict) + (look_back * 2) - 1:len(dataset2) - 1, :] = testPredict

        # plot baseline and predictions

        plt.plot(scaler.inverse_transform(dataset2))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.title("LSTM con %i neuronas, correlacion: %.2f" % (neuronas, corr[1][0] * 100))
        plt.ylabel("Flujo Sanguineo Cerebral")
        plt.xlabel("Tiempo")
        plt.show()

        registro = pandas.read_csv('training.log', sep=",")
        plt.plot(registro["epoch"].values, registro["loss"].values)
        plt.title("Error LSTM con %i neuronas, correlacion: %.2f" % (neuronas, corr[1][0] * 100))
        plt.ylabel("Error")
        plt.xlabel("Epocas")
        plt.show()


parser = argparse.ArgumentParser(description='Ejecucion del dataSet con LSTM')
parser.add_argument('-n','--neuronas', help='Numero de neuronas en la capa oculta',required=True, type=int)
parser.add_argument('-c','--ciclos',help='Numero de epocas de la LSTM', required=True, type=int)
parser.add_argument('-a','--activacion', help='Escoger funcion de activacion. 0: sigmoid, 1: relu',required=True, type=int)
parser.add_argument('-l','--lookBack',help='Cantidad de peridos hacia atras que son necesarios para predecir el futuro', required=True, type=int)
parser.add_argument('-f','--folds',help='Cantidad de folds en la validacion cruzada', required=True, type=int)

args = parser.parse_args()

cantidad = 3
dataframe = pandas.read_csv('HC092101.PAR', usecols=[3], engine='python', skipfooter=3, header=None)
dataframe = remuestre(cantidad,3, dataframe)
dataset1 = dataframe.values
dataset1 = dataset1.astype('float32')

dataframe = pandas.read_csv('HC092101.PAR', usecols=[2], engine='python', skipfooter=3, header=None)
dataframe = remuestre(cantidad, 2,dataframe)
dataset2 = dataframe.values
dataset2 = dataset2.astype('float32')

# normalizacion del dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(dataset1)
dataset2 = scaler.fit_transform(dataset2)

cross_validation(dataset1, dataset2, args.folds, args.activacion, args.ciclos, args.neuronas, args.lookBack)