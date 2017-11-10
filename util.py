import numpy as np

def cantidadDeNeuronas(data):
    """Funcion que permite calcular la cantidad de neuronas a utilizar
    :param data: Datos utilizados
    :return: Cantidad de neuronas a utilizar
    """
    ne = 3
    ns = 1
    nw = len(data)/10
    nc = int((nw-ns)/(ne+ns+1))
    return nc

def normalizar(data):
    """Funcion que permite normalizar los datos entre 0 y 1
    :param data: Datos a ser normalizados
    :return: retorna los datos normalizados
    """
    max = np.max(data)
    min = np.min(data)
    data = (data-min)/(max-min)
    return data

#Funcion para realizar el remuestro de la senal
#Entrada:   data_pacientes: Datos del paciente a remuestrear
#           remuestreo: cantidad de muestras a considerar para realizar la nueva muestra
def remuestreo(data_pacientes, remuestreo):
    """Funcion que permite realizar el remuestreo de los datos
    :param data_pacientes: datos originales
    :param remuestreo: tasa de remuestreo
    :return: datos remuestreados
    """
    abp = np.array(data_pacientes["ABP"])
    cbfv = np.array(data_pacientes["CBFV.C1"])
    abp1 = np.array([])
    cbfv1 = np.array([])
    for i in range(len(abp)):
        if (i % remuestreo == 0 and i != len(abp) - 1):
            newabp1 = np.sum(abp[i:i + remuestreo]) / remuestreo
            # print("Promedio de abp ", abp[i:i+remuestreo])
            newcbfv1 = np.sum(cbfv[i:i + remuestreo]) / remuestreo
            abp1 = np.append(abp1, newabp1)
            cbfv1 = np.append(cbfv1, newcbfv1)
    abp1 = normalizar(abp1)
    cbfv1 = normalizar(cbfv1)
    return abp1, cbfv1

#Se aplican por defectos tres retrasos
#Entrada: Datos de la senal a ser retrasada
def retrasos(data):
    """Funcion que permite aplicar retrasos a la funcion

    :param data: serie de tiempo a retrasar, en este caso ABP
    :return: retorna el arregla con los retrasos aplicados
    """
    array1 = []
    array2 = []
    array3 = []
    for i in range(0,len(data)-3):
        array1.append(data[i])
    for i in range(1,len(data)-2):
        array2.append(data[i])
    for i in range(2,len(data)-1):
        array3.append(data[i])
    array = [array1,array2,array3]
    array = np.array(array)
    return array.T


