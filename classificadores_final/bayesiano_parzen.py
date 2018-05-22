from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV

def estimar_parametros(dados, rotulos): # retorna o nome das classes e a probabilidade a priori de cada uma
    nbg = GaussianNB()
    nbg.fit(dados, rotulos)
    classes = nbg.classes_
    p_priori = nbg.class_prior_
    return classes, p_priori

def estimar_bandwidth(dados):
    # criar dicionário para gerar largura de banda
    parametros = {'bandwidth': np.logspace(0.3, 1, 10)}
    grid = GridSearchCV(KernelDensity(), parametros, cv=2) #cv = usando StratifiedKFold
    grid.fit(dados)
    return grid.best_estimator_.bandwidth

# separa o conjunto de treinamento em subconjuntos por classe; recebe como parâmetro o conjunto de treinamento (x) e conjunto de rótulos de treinamento (y)
# para cada subconjunto de dados é estimado o Kernel Density (por classe)
def dados_por_classe(x, y, classes):
    func_densidades = []
    h = estimar_bandwidth(x)
    for i in range(len(classes)):
        dados_classe = []
        for j in range(len(y)):
            if (y[j] == classes[i]):
                dados_classe.append(x[j])
        func_densidades.append(estimar_kde(dados_classe, h)) # adiciona em um vetor a densidade de kernel para cada classe
    return func_densidades

# estima a densidade kernel - recebe os dados  e uma janela de parzen
def estimar_kde(dados, janela):
    kde = KernelDensity(kernel='gaussian', bandwidth=janela).fit(dados)
    return kde

def calcular_densidade(amostra, func_densidades, classes, p_priori): # para cada amostra é calculado a densidade de cada classe
    densidades = []
    amostra = (amostra.reshape((1, amostra.shape[0])))  # ajuste no formato da amostra
    for i in range(len(classes)):
        res = np.exp( (func_densidades[i].score_samples(amostra)) + np.log(p_priori[i])) # incluindo a prob a priori
        densidades.append(res)
    return densidades

def calcular_posteriori(densidades, classes): # para cada amostra é calculado a prob a posteriori de cada classe
    evidencia = sum(densidades) # calculando a evidência (soma das densidades de cada classe)
    p_posteriori = []
    for i in range(len(classes)):
        p_posteriori.append(densidades[i]/evidencia)
    return p_posteriori

def max_posteriori(p_posteriori):
    posteriori = max(p_posteriori)
    return posteriori

def escolha_classe(p_posteriori, classes):
    return classes[p_posteriori.index(max(p_posteriori))]

def executar_parzen(dados_treinamento, rotulos_treinamento, dados_teste):
    parametros = estimar_parametros(dados_treinamento, rotulos_treinamento)
    # retorna a função de densidade de cada classe já com a janela h estimada
    func_densidades = dados_por_classe(dados_treinamento, rotulos_treinamento, parametros[0])
    posteriori_parzen = []
    for amostra in dados_teste: # para cada amostra do conjunto de teste...
         densidades = calcular_densidade(amostra, func_densidades, parametros[0], parametros[1]) # calcula a densidade de cada classe + prob. a priori (log)
         posteriori = calcular_posteriori(densidades, parametros[0]) # calcula a posteriori de cada classe + evidência
         post = []
         for i in range(len(posteriori)): # ajuste para que o vetor de posteriori não saia no formato nparray
             post.append(posteriori[i][0])
         posteriori_parzen.append(post)
    return posteriori_parzen # devolve uma matriz c x n, com a probabilidade a posteriori de cada classe (c) com n amostras
