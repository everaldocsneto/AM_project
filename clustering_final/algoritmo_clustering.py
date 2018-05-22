import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score

# calcula o valor de gama
def calcular_gama(dados):
    distancias = euclidean_distances(dados, dados, squared=True)  # calculando distância euclidiana ao quadrado
    lista_distancias = np.sort(distancias[np.triu_indices(len(distancias), 1)]) # transformando a matriz em uma lista e ordena os valores

    indice_quantil_1 = int((len(lista_distancias) * (1 / 10)) - 1)  # calculando o índice do quantil 0.1
    indice_quantil_9 = int((len(lista_distancias) * (9 / 10)) - 1)  # calculando o índice do quantil 0.9

    sigma = (lista_distancias[indice_quantil_1] + lista_distancias[indice_quantil_9]) / 2 # média do quantil 0.1 e 0.9
    gama = (1/sigma)
    return gama

# inicializa o vetor de hiperparametros com o valor de gama
def iniciar_hyperparametros(dados, gama):
    hp = []
    for i in range(dados.shape[1]):
        hp.append(gama)
    return hp

# inicializa o vetor de protótipos randomicamente
def iniciar_prototipos(c, dados):
    prototipos = []
    for i in range(c):
        indice = np.random.randint(0, len(dados)- 1)
        prototipos.append(dados[indice])
    return prototipos

# inicializa c clusters vazios
def iniciar_clusters(c):
    clusters = []
    for i in range(c):
        clusters.append([])
    return clusters

# atribuição inicial dos objetos ao cluster
def iniciar_afetacao_objeto(c, dados, v_prototipos, v_hp):
    clusters = iniciar_clusters(c)
    for i in range(len(dados)):
        resultados = [] # para cada objeto será calculada a variante KCM-K-GH considerando cada protótipo...
        for p in range(len(v_prototipos)):
            resultados.append(2*(1 - calcular_variante_K(dados[i], v_prototipos[p], v_hp)))
        indice_menor_distancia = resultados.index(min(resultados)) # o objeto será atribuido ao cluster que possuir a menor distância
        clusters[indice_menor_distancia].append(i)
    return clusters

############# EQUAÇÕES #############

# calcula a variante KCM-K-GH para um objeto x, dado um protótipo p e um vetor de hiperparametros
def calcular_variante_K(x, p, hp): # Equação 9
    distancia = 0
    for j in range(len(hp)):
        distancia += hp[j] * np.power((x[j]-p[j]), 2)
    resultado = np.exp((-1.0/2.0)*distancia)
    return resultado

# calcula o representante do cluster (novos protótipos)
def calcular_representantes_clusters(dados, v_cluster, v_prototipos, v_hp):  # Equação 14
    for c in range(len(v_prototipos)):
        if (len(v_cluster[c]) > 0):
            soma1, soma2 = 0, 0
            for e in range(len(v_cluster[c])):
                xk = dados[v_cluster[c][e]] # acessando o objeto xk
                gi = v_prototipos[c] # acessando o protótipo
                soma1 += (calcular_variante_K(xk, gi, v_hp)) * xk
                soma2 += (calcular_variante_K(xk, gi, v_hp))
            v_prototipos[c] = soma1 / soma2
    return v_prototipos

# calcula o vetor de hyperparametros
def calcular_hyperparametros(dados, v_prototipos, v_clusters, gama, v_hp):  # Equação 16
    hparametros = []
    produto = 1

    for h in range(dados.shape[1]):
        soma1 = 0
        for c in range(len(v_clusters)):
            soma2 = 0
            for k in range(len(v_clusters[c])):
                xk = dados[v_clusters[c][k]]
                gi = v_prototipos[c]
                parte1 = calcular_variante_K(xk, gi, v_hp)
                parte2 = np.power(xk[h] - gi[h], 2)
                soma2 += parte1 * parte2
            soma1 += soma2
        produto *= soma1
    resultado1 = np.power(produto, 1 / dados.shape[1]) * gama

    for j in range(dados.shape[1]):
        # calculando resultado 2
        soma1 = 0
        for w in range(len(v_clusters)):
            soma2 = 0
            for k in range(len(v_clusters[w])):
                xk = dados[v_clusters[w][k]]
                gi = v_prototipos[w]
                parte1 = calcular_variante_K(xk, gi, v_hp)
                parte2 = np.power(xk[j] - gi[j], 2)
                soma2 += parte1 * parte2
            soma1 += soma2
        resultado2 = soma1
        hparametros.append(resultado1 /resultado2)
    return hparametros

# para cada objeto x do conjunto de dados...
def atribuir_objeto_cluster(x, v_prototipos, v_hp): # etapa de alocação - Equação 18
    resultado = []
    for p in range(len(v_prototipos)):
        resultado.append(2*(1-calcular_variante_K((x), v_prototipos[p], v_hp)))
    indice_distancia_menor = resultado.index(min(resultado))
    return indice_distancia_menor # retornar o índice da menor distância

# busca o cluster atual de um objeto
# recebe o indice de um objeto x do conjunto de dados e uma lista c/ os clusters
def retorna_indice_atual(i, clusters):
    indice_atual = 0
    for c in range(len(clusters)):
        if ((i in clusters[c]) == True):
            indice_atual = c
            break
    return indice_atual

# calcula a função objetivo para um cluster gerado
def funcao_objetivo(dados, v_clusters, v_prototipos, v_hp): # Equação 11
    resultado = 0
    for i in range(len(v_clusters)):
        for k in range(len(v_clusters[i])):
            resultado += 2*(1 - calcular_variante_K(dados[v_clusters[i][k]] , v_prototipos[i],v_hp))
    return resultado


# calcula o indice de Rand ajustado
# gera um vetor com os índices do cluster e um vetor com os rótulos de classe
def indice_rand(dados, rotulos, v_clusters):
    labels_pred = []
    labels_true = []
    for i in range(len(dados)):
        for c in range(len(v_clusters)):
            if (i in v_clusters[c]) == True:
                labels_pred.append(c)
                break
    for j in range(len(rotulos)):
        for h in range(len(rotulos[j])):
            labels_true.append(rotulos[j][h])
    return adjusted_rand_score(labels_true, labels_pred)