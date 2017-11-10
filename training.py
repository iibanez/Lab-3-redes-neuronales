# coding=utf-8
from sklearn.neural_network import MLPRegressor
from util import retrasos, cantidadDeNeuronas
import numpy as np
def k2(ABP,CBFV):
    """Funcion utilizada para crear dos grupos de validacion

    :param ABP: Serie de senal de presion arterial media
    :param CBFV: Serie de senal de velocidad de flujo cerebral
    :return:
    """
    ABPA = ABP[0:int(len(ABP)/2),]
    CBFVA = CBFV[0:int(len(CBFV)/2)-3,]

    ABPB = ABP[int(len(ABP)/2):len(ABP),]
    CBFVB = CBFV[int(len(CBFV)/2):len(CBFV)-3,]
    return retrasos(ABPA),retrasos(ABPB),CBFVA,CBFVB

def k3(ABP,CBFV):
    """Funcion utilizada para crear tres grupos de validacion

    :param ABP: Serie de senal de presion arterial media
    :param CBFV: Serie de senal de velocidad de flujo cerebral
    :return:
    """
    k = 3
    ABPA = ABP[0:int(len(ABP)/k),]
    CBFVA = CBFV[0:int(len(CBFV)/k)-3,]

    ABPB = ABP[int(len(ABP)/k):int(2*len(ABP)/k),]
    CBFVB = CBFV[int(len(CBFV)/k):int(2*len(CBFV)/k)-3,]

    ABPC = ABP[int(2*len(ABP)/k):len(ABP),]
    CBFVC = CBFV[int(2*len(CBFV)/k)-3:len(CBFV)-6,]
    return retrasos(ABPA),CBFVA,retrasos(ABPB),CBFVB,retrasos(ABPC),CBFVC

def k4(ABP,CBFV):
    """Funcion utilizada para crear los grupos para la validacion cruzada

    :param ABP: Datos de presion arterial media
    :param CBFV: Datos para velocidad de flujo cerebral
    :return: Conjunto de datos
    """
    k=4
    #Obtencion de los conjuntos de test y training
    ABPA = ABP[0:int(len(ABP)/k),]
    CBFVA = CBFV[0:int(len(CBFV)/k),]

    ABPB = ABP[int(len(ABP)/k):int(2*len(ABP)/k),]
    CBFVB = CBFV[int(len(CBFV)/k):int(2*len(CBFV)/k),]

    ABPC = ABP[int(2*len(ABP)/k):int(3*len(ABP)/k),]
    CBFVC = CBFV[int(2*len(CBFV)/k):int(3*len(CBFV)/k),]

    ABPD = ABP[int(3*len(ABP)/k):len(ABP),]
    CBFVD = CBFV[int(3*len(CBFV)/k):len(CBFV),]

    ABPA = retrasos(ABPA)
    ABPB = retrasos(ABPB)
    ABPC = retrasos(ABPC)
    ABPD = retrasos(ABPD)

    CBFVA = CBFVA[0:len(CBFVA)-np.abs(len(CBFVA)-len(ABPA)),]
    CBFVB = CBFVB[0:len(CBFVB)-np.abs(len(CBFVB)-len(ABPB)),]
    CBFVC = CBFVC[0:len(CBFVC)-np.abs(len(CBFVC)-len(ABPC)),]
    CBFVD = CBFVD[0:len(CBFVD)-np.abs(len(CBFVD)-len(ABPD)),]
    return ABPA,CBFVA,ABPB,CBFVB,ABPC,CBFVC,ABPD,CBFVD

def train(X,Y,n):
    """Funcion de entrenamiento de una red neuronal
    :param X: Datos de entrenamiento para la entrada
    :param Y: Datos de entrenamiento de salida
    :param n: Cantidad máxima de iteraciones
    :return: Retona una red neuronal con los pesos ajustados
    """
    mpl = MLPRegressor(hidden_layer_sizes=cantidadDeNeuronas(X),activation="relu",solver="sgd",alpha=0.01,learning_rate="constant",verbose=False,early_stopping=True,max_iter=n)
    mpl.fit(X,Y)
    return mpl
