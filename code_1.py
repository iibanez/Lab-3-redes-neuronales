import numpy as np
import pandas as pd
from util import remuestreo, retrasos
from training import k2, k3, k4, train
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys
import matplotlib.pyplot as plt
from optimizacion import gridSearch
import argparse
import scipy.stats.stats as sp
from scipy import fftpack as fft
import random

random.seed(10)
head = ["id","Time", "CBFV.C1", "ABP", "CBFV.C2", "End-Tindal", "CrCP.C1", "RAP.C1", "Heart Rate", "Sistolic Pressure",
        "Diastolic Pressure", "POURCELOT'S INDEX  CHANNEL 1", "MEAN SQUARE ERROR FOR P-V CH. 1", "CrCP.C2", "RAP.C2",
        "SYSTOLIC  VEL. CH.1", "DIASTOLIC VEL. CH.1", "SYSTOLIC VELOCITY CH. 2", "DIASTOLIC VELOCITY CH.2",
        "POURCELOT INDEX CH. 2", "MSE FOR P-V  CH. 2"]


def generalizacion(modelo, entrada, salida_esperada, titulo, model_number, filename):
    """Permite revisar la generalizacion del modelo utilizado.

    :param modelo: Modelo ya entrenado
    :param entrada: Valores de entreada utilizados
    :param salida_esperada: Valores de salida utilizados
    :param titulo: Titulo del grafico
    :param model_number: Numero de modelo
    :param filename: Nombre de archivo a guardar
    :return: Retorna el error obtenido a demas de guardar el grafico de la senal predicha
    """
    generalizacion1 = modelo.predict(entrada)
    mserg = mean_squared_error(salida_esperada, generalizacion1)
    maerg = mean_absolute_error(salida_esperada, generalizacion1)
    corrg = np.corrcoef(salida_esperada,generalizacion1)
    print("Con el modelo %d se obtiene un mser de %f, un maer de %f  y una correlacion de %f generalizando" % (model_number, mserg, maerg,corrg[1][0]))
    fig, (ax_orig, ax_predicted) = plt.subplots(2, 1)
    ax_orig.plot(salida_esperada, "r")
    ax_predicted.plot(generalizacion1, "b")
    ax_orig.set_title("Original Signal")
    ax_predicted.set_title(titulo)
    plt.savefig(filename)
    plt.show()
    residual_power = salida_esperada-generalizacion1
    frq = fft.fftfreq(len(residual_power),1/0.6)
    plt.plot(frq,abs(fft.fft(residual_power)))
    plt.xlabel('Frequency Hz')
    plt.ylabel('|Residual power|')
    plt.show()
    return mserg

#Se considera un validacion cruzada con k = 2
def cvK2(ABP, CBFV, ABP_G, CBFV_G, tipe,n):
    """Funcion que permite generar la validacion cruzada con dos pliegues
    :param ABP: Senal de abp para entrenamiento, es dividido en entrenamiento y test
    :param CBFV: Senal de cbfv para el entrenamiento, es dividido en entrenamiento y test
    :param ABP_G: Senal utilizada para la generalizacion
    :param CBFV_G: Senal utilizada para la generalizacoin
    :param tipe: Dice si se utiliza una senal en hipercapnia (1) o normocapnia (0)
    :param n: Cantidad de iteracioens maximas para la red
    :return:
    """
    k=2
    #
    # Obtencion de los conjuntos de test y training
    pacienteNABPA, pacienteNABPB, pacienteNCBFVA, pacienteNCBFVB = k2(ABP, CBFV)

    #
    # Se entrena con A, se testea con B
    mpl_k1 = train(pacienteNABPA, pacienteNCBFVA,n)
    pacienteNCBFVEstimated = mpl_k1.predict(pacienteNABPB)
    mser = mean_squared_error(pacienteNCBFVB, pacienteNCBFVEstimated)
    maer = mean_absolute_error(pacienteNCBFVB, pacienteNCBFVEstimated)
    corr = np.corrcoef(pacienteNCBFVB,pacienteNCBFVEstimated)
    print("Con el modelo 1 se obtiene un mser de %f, un maer de %f y una correlacion de %f testeando" % (mser, maer,corr[1][0]))

    #
    # Se entrena con B, se testea con A
    mpl_k2 = train(pacienteNABPB, pacienteNCBFVB,n)
    pacienteNCBFVEstimated = mpl_k2.predict(pacienteNABPA)
    mser = mean_squared_error(pacienteNCBFVA, pacienteNCBFVEstimated)
    maer = mean_absolute_error(pacienteNCBFVA, pacienteNCBFVEstimated)
    corr = np.corrcoef(pacienteNCBFVA,pacienteNCBFVEstimated)
    print("Con el modelo 2 se obtiene un mser de %f, un maer de %f y una correlacion de %f testeando" % (mser, maer,corr[1][0]))

    #
    # GENERALIZACION
    mser1 = generalizacion(mpl_k1, ABP_G, CBFV_G, "Model: 1, Train: A, Test: B", 1, tipe+"model1K2.png")
    mser2 = generalizacion(mpl_k2, ABP_G, CBFV_G, "Model: 2, Train: B, Test: A", 2, tipe+"model2K2.png")
    if mser1 > mser2:
        return mpl_k2
    else:
        return mpl_k1


######################### ENTRENAMIENTO 1################################

def cvk3(ABP, CBFV, ABP_G, CBFV_G, tipe, n):
    """Funcion que permite generar la validacion cruzada con tres pliegues
    :param ABP: Senal de abp para entrenamiento, es dividido en entrenamiento y test
    :param CBFV: Senal de cbfv para el entrenamiento, es dividido en entrenamiento y test
    :param ABP_G: Senal utilizada para la generalizacion
    :param CBFV_G: Senal utilizada para la generalizacoin
    :param tipe: Dice si se utiliza una senal en hipercapnia (1) o normocapnia (0)
    :param n: Cantidad de iteracioens maximas para entrenamiento de la red
    :return:
    """
    #Se considera un validacion cruzada con k = 3
    k=3
    #Obtencion de los conjuntos de test y training
    pacienteNABPA, pacienteNCBFVA, pacienteNABPB, pacienteNCBFVB, pacienteNABPC, pacienteNCBFVC = k3(ABP, CBFV)
    #Se aplican los retrasos a los conjuntos de test y de training

    #Se entrena con a y b, y se entrena con c
    mpl_k1 = train(np.concatenate([pacienteNABPA, pacienteNABPB]), np.concatenate([pacienteNCBFVA, pacienteNCBFVB]),n)
    pacienteNCBFVEstimated = mpl_k1.predict(pacienteNABPC)
    mser = mean_squared_error(pacienteNCBFVC, pacienteNCBFVEstimated)
    maer = mean_absolute_error(pacienteNCBFVC, pacienteNCBFVEstimated)
    corr = np.corrcoef(pacienteNCBFVC,pacienteNCBFVEstimated)
    print("Con el modelo 1 se obtiene un mser de %f, un maer de %f y una correlacion de %f testeando" % (mser, maer,corr[1][0]))

    #Se entrena con a y c, y se prueba con b
    mpl_k2 = train(np.concatenate([pacienteNABPA, pacienteNABPC]), np.concatenate([pacienteNCBFVA, pacienteNCBFVC]),n)
    pacienteNCBFVEstimated = mpl_k2.predict(pacienteNABPB)
    mser = mean_squared_error(pacienteNCBFVB, pacienteNCBFVEstimated)
    maer = mean_absolute_error(pacienteNCBFVB, pacienteNCBFVEstimated)
    corr = np.corrcoef(pacienteNCBFVB,pacienteNCBFVEstimated)
    print("Con el modelo 2 se obtiene un mser de %f, un maer de %f y una correlacion de %f testeando" % (mser, maer,corr[1][0]))

    #Se entrena con b y c, y se entrena con a
    mpl_k3 = train(np.concatenate([pacienteNABPB, pacienteNABPC]), np.concatenate([pacienteNCBFVB, pacienteNCBFVC]),n)
    pacienteNCBFVEstimated = mpl_k3.predict(pacienteNABPA)
    mser = mean_squared_error(pacienteNCBFVA, pacienteNCBFVEstimated)
    maer = mean_absolute_error(pacienteNCBFVA, pacienteNCBFVEstimated)
    corr = np.corrcoef(pacienteNCBFVA,pacienteNCBFVEstimated)
    print("Con el modelo 3 se obtiene un mser de %f, un maer de %f y una correlacion de %f testeando" % (mser, maer,corr[1][0]))

    #GENERALIZACION
    mser1 = generalizacion(mpl_k1, ABP_G, CBFV_G, "Model: 1, Train: A-B, Test: C", 1, tipe+"model1K3.png")
    mser2 = generalizacion(mpl_k2, ABP_G, CBFV_G, "Model: 2, Train: A-C, Test: B", 2, tipe+"model2K3.png")
    mser3 = generalizacion(mpl_k3, ABP_G, CBFV_G, "Model: 3, Train: B-C, Test: A", 3, tipe+"model3K3.png")
    if mser1 == np.min(np.array([mser1, mser2, mser3])):
        return mpl_k1
    if mser2 == np.min(np.array([mser1, mser2, mser3])):
        return mpl_k2
    if mser3 == np.min(np.array([mser1, mser2, mser3])):
        return mpl_k3


# Captura de datos desde csv
# Lectura de los datos de los sujetos en Hipercapnia

def cvk4(ABP, CBFV, ABP_G, CBFV_G, tipe,n):
    """Funcion que permite generar la validacion cruzada con cuatro pliegues
    :param ABP: Senal de abp para entrenamiento, es dividido en entrenamiento y test
    :param CBFV: Senal de cbfv para el entrenamiento, es dividido en entrenamiento y test
    :param ABP_G: Senal utilizada para la generalizacion
    :param CBFV_G: Senal utilizada para la generalizacoin
    :param tipe: Dice si se utiliza una senal en hipercapnia (0) o normocapnia (1)
    :param n: Cantidad de iteracioens de entrenamiento maximasr
    :return:
    """
    #Se considera un validacion cruzada con k = 4
    pacienteNABPA, pacienteNCBFVA, pacienteNABPB, pacienteNCBFVB, pacienteNABPC, pacienteNCBFVC, pacienteNABPD, pacienteNCBFVD = k4(ABP, CBFV)

    #Se entrena con a,b y c, se prueba con d
    mpl_k1 = train(np.concatenate([pacienteNABPA, pacienteNABPB, pacienteNABPC]), np.concatenate([pacienteNCBFVA, pacienteNCBFVB, pacienteNCBFVC]),n)
    pacienteNCBFVEstimated = mpl_k1.predict(pacienteNABPD)
    mser = mean_squared_error(pacienteNCBFVD, pacienteNCBFVEstimated)
    maer = mean_absolute_error(pacienteNCBFVD, pacienteNCBFVEstimated)
    corr = np.corrcoef(pacienteNCBFVD,pacienteNCBFVEstimated)
    print("Con el modelo 1 se obtiene un mser de %f, un maer de %f y una correlacion de %f testeando" % (mser, maer,corr[1][0]))

    #Se entrena con a,b y d, se prueba con c
    mpl_k2 = train(np.concatenate([pacienteNABPA, pacienteNABPB, pacienteNABPD]), np.concatenate([pacienteNCBFVA, pacienteNCBFVB, pacienteNCBFVD]),n)
    pacienteNCBFVEstimated = mpl_k2.predict(pacienteNABPC)
    mser = mean_squared_error(pacienteNCBFVC, pacienteNCBFVEstimated)
    maer = mean_absolute_error(pacienteNCBFVC, pacienteNCBFVEstimated)
    corr = np.corrcoef(pacienteNCBFVC,pacienteNCBFVEstimated)
    print("Con el modelo 2 se obtiene un mser de %f, un maer de %f y una correlacion de %f testeando" % (mser, maer,corr[1][0]))

    #Se entrena con a,c y d, se prueba con b
    mpl_k3 = train(np.concatenate([pacienteNABPA, pacienteNABPC, pacienteNABPD]), np.concatenate([pacienteNCBFVA, pacienteNCBFVC, pacienteNCBFVD]),n)
    pacienteNCBFVEstimated = mpl_k3.predict(pacienteNABPD)
    mser = mean_squared_error(pacienteNCBFVB, pacienteNCBFVEstimated)
    maer = mean_absolute_error(pacienteNCBFVB, pacienteNCBFVEstimated)
    corr = np.corrcoef(pacienteNCBFVB,pacienteNCBFVEstimated)
    print("Con el modelo 3 se obtiene un mser de %f, un maer de %f y una correlacion de %f testeando" % (mser, maer,corr[1][0]))

    #Se entrena con b,c y d, se prueba con a
    mpl_k4 = train(np.concatenate([pacienteNABPB, pacienteNABPC, pacienteNABPD]), np.concatenate([pacienteNCBFVB, pacienteNCBFVC, pacienteNCBFVD]),n)
    pacienteNCBFVEstimated = mpl_k4.predict(pacienteNABPA)
    mser = mean_squared_error(pacienteNCBFVA, pacienteNCBFVEstimated)
    maer = mean_absolute_error(pacienteNCBFVA, pacienteNCBFVEstimated)
    corr = np.corrcoef(pacienteNCBFVA,pacienteNCBFVEstimated)
    print("Con el modelo 4 se obtiene un mser de %f, un maer de %f y una correlacion de %f testeando" % (mser, maer,corr[1][0]))
    #GENERALIZACION
    mser1 = generalizacion(mpl_k1, ABP_G, CBFV_G, "Model: 1, Train: A-B-C, Test: D", 1, tipe+"model1K4.png")
    mser2 = generalizacion(mpl_k2, ABP_G, CBFV_G, "Model: 2, Train: A-B-D, Test: C", 2, tipe+"model2K4.png")
    mser3 = generalizacion(mpl_k3, ABP_G, CBFV_G, "Model: 3, Train: A-C-D, Test: B", 3, tipe+"model3K4.png")
    mser4 = generalizacion(mpl_k4, ABP_G, CBFV_G, "Model: 4, Train: B-C-D, Test: A", 4, tipe+"model4K4.png")

    if mser1 == np.min(np.array([mser1, mser2, mser3, mser4])):
        return mpl_k1
    elif mser2 == np.min(np.array([mser1, mser2, mser3, mser4])):
        return mpl_k2
    elif mser3 == np.min(np.array([mser1, mser2, mser3, mser4])):
        return mpl_k3
    elif mser4 == np.min(np.array([mser1, mser2, mser3, mser4])):
        return mpl_k4
    return

def neural_network(ABP, CBFV,ABP_G, CBFV_G,tipe):
    #Mejores modelos obtenidos con cross validation

    print("\t\tCross validation K = 2")
    #
    print("\t\tCross validation K = 3")
    #mplk3 = cvk3(ABP, CBFV,ABP_G, CBFV_G,tipe)
    print("\t\tCross validation K = 4")
    mplk4 = cvk4(ABP, CBFV,ABP_G, CBFV_G,tipe)

    return
#####################################################
##      EJECUCION DE PROGRAMA                      ##
#####################################################

# Captura de datos desde csv
# Lectura de los datos de los sujetos en Hipercapnia


parser = argparse.ArgumentParser()
parser.add_argument("condition",type = int, help="Determina si se entrena con hipercapnia(0) o normocapnia(1)")
parser.add_argument("fold",type = int,help="Determina la cantidad de pliegues a usar = 2,3 o 4")
parser.add_argument("-ft","--fileTraining", help="Ubicacion de archivo con datos de entrenamiento")
parser.add_argument("-fg","--fileGeneralization", help="Ubicacion de archivo con datos de generalizacion")
parser.add_argument("-op","--optimization", help = "Muestra los hiperparametros optimizados", action = "store_true")
parser.add_argument("epoch",type=int,help="Cantidad de iteraciones de entrenamiento de red neuronal")
args = parser.parse_args()

try:
    pacientT = pd.read_csv(args.fileTraining, sep=";", header=None, names=head)
    pacienteTABP, pacienteTCBFV = remuestreo(pacientT,3) #Los retrasos son aplicados dentro de las funciones
    print("Son cargados los datos de entrenamiento")
    pacienteG = pd.read_csv(args.fileGeneralization, sep=";", header=None, names=head)
    pacienteGABP, pacienteGCBFV = remuestreo(pacienteG,3)
    pacienteGABP = retrasos(pacienteGABP)
    pacienteGCBFV = pacienteGCBFV[0:len(pacienteGCBFV)-3,]
    print("Son cargados los datos de generalizacion")
    if args.optimization:
        print("Optimizacion")
        model, mser, params = gridSearch(pacienteTABP, pacienteTCBFV ,pacienteGABP, pacienteGCBFV)
        print(model)
        print(mser)
        print(params)
    if args.fold == 2 :
        print("\t\tCross validation K = 2")
        if args.condition == 0 :
            mplk2 = cvK2(pacienteTABP, pacienteTCBFV,pacienteGABP, pacienteGCBFV,"hiper",args.epoch)
        elif args.condition == 1 :
            mplk2 = cvK2(pacienteTABP, pacienteTCBFV,pacienteGABP, pacienteGCBFV,"normo",args.epoch)
        else:
            print("Opcion no reconocida")
    elif args.fold == 3:
        print("\t\tCross validation K = 3")
        if args.condition == 0:
            mplk3 = cvk3(pacienteTABP, pacienteTCBFV,pacienteGABP, pacienteGCBFV,"hiper",args.epoch)
        elif args.condition == 1:
            mplk3 = cvk3(pacienteTABP, pacienteTCBFV,pacienteGABP, pacienteGCBFV,"normo",args.epoch)
        else:
            print("Opcion no reconocida")
    elif args.fold == 4:
        print("\t\tCross validation K = 4")
        if args.condition == 0:
            mplk4 = cvk4(pacienteTABP, pacienteTCBFV, pacienteGABP, pacienteGCBFV, "hiper",args.epoch)
        elif args.condition == 1:
            mplk4 = cvk4(pacienteTABP, pacienteTCBFV, pacienteGABP, pacienteGCBFV, "normo",args.epoch)
        else:
            print("Opcion no reconocida")
    else:
        print("Utilizar 2, 3 o 4 pliegues")
            
except (IOError):
        print("Error archivos no existentes, por favor revisar archivoss.")
        
