from sklearn.naive_bayes import GaussianNB
import numpy as np
from numpy.linalg import inv, det

def estimar_parametros(dados, rotulos):
    nbg = GaussianNB()
    nbg.fit(dados, rotulos)
    clss = nbg.classes_
    mi = nbg.theta_
    variancia = nbg.sigma_
    p_priori = nbg.class_prior_
    return clss, mi, variancia, p_priori

def calcular_verossimilhanca(x, parametros): # cada amostra calcula a verossimilhança
    clss = parametros[0]
    mi = parametros[1]
    variancia = parametros[2]
    p_priori = parametros[3]
    verossimilhanca = []
    for c in range(len(clss)): # para cada classe
        resultado = 0
        mtx_inv = inv(np.identity(x.shape[0]) * variancia[c])  # matriz inversa da variância
        calc1 = np.power(2 * np.pi, -(x.shape[0] / 2)) * (np.power(det(mtx_inv), 1 / 2))
        diff = x - mi[c]
        first = np.dot(diff.reshape((1, x.shape[0])), mtx_inv)
        calc2 = np.exp((-1 / 2 * (np.dot(first, diff.reshape((x.shape[0]), 1)))))
        resultado = float(calc1 * calc2)
        verossimilhanca.append(resultado * p_priori[c])  # faz a ponderação com a prob a priori
    return verossimilhanca

def calcular_posteriori(verossimilhanca): # para cada amostra calcula a posteriori
    posteriori = []
    evidencia = sum(verossimilhanca)
    for i in range(0, len(verossimilhanca)):
        posteriori.append(verossimilhanca[i] / evidencia)
    return posteriori

def executar_gauss(dados_treinamento, rotulos_treinamento, dados_teste): # para o conjunto de teste...
    parametros = estimar_parametros(dados_treinamento, rotulos_treinamento) # estima os parâmetros com base no conjunto de treinamento
    posteriori_gauss = []
    for amostra in dados_teste: # para cada nova amostra do conjunto de teste...
        verossimilhanca = calcular_verossimilhanca(amostra, parametros) # calcula a verossimihança
        posteriori = calcular_posteriori(verossimilhanca) # calcula a probabilidade a posteriori, incluindo a evidência
        posteriori_gauss.append(posteriori)
    return posteriori_gauss # devolve uma matriz c x n, com a probabilidade a posteriori de cada classe (c) com n amostras