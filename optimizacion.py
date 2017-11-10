from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from util import cantidadDeNeuronas,retrasos


def gridSearch(ABP,CBFV,ABP_G,CBFV_G):
    """
    Funcion utilizada para determiar los mejores parametros de la red
    :param ABP: senal de presion arterial media
    :param CBFV: senal de velocidad de flujo sanguine cerebral
    :param ABP_G: senal de presion arterial media
    :param CBFV_G: senal de velocidad de flujo sanguine cerebral
    :return: Retornal el error de la red optimizada y los parametros a utilizar
    """
    #Parametros a utilizar
    learn_rate = [0.001,0.003,0.01,0.03,0.1,0.3]
    hidden_layer_sizes = [2,3,4,5,6,cantidadDeNeuronas(retrasos(ABP))]
    activation = ["identity","logistic", "tanh", "relu"]

    #Modelo obtenidocon Optimizacion de parametros
    model = MLPRegressor(max_iter=10000)
    grid = GridSearchCV(estimator=model, param_grid=dict(activation = activation,alpha=learn_rate, hidden_layer_sizes = hidden_layer_sizes))
    CBFV = CBFV[0:len(CBFV)-3]
    grid.fit(retrasos(ABP),CBFV)
    model = grid.best_estimator_
    params = grid.best_params_
    CBFV_G1 = model.predict(ABP_G)
    mser = mean_squared_error(CBFV_G,CBFV_G1)
    return model, mser, params
