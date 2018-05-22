import clustering_final.algoritmo_clustering as code
import pandas as pd
import time

# Variáveis para armazenar resultado da execução do algoritmo
res_func_objetivo = []
res_cluster = []
res_hp = []
res_prototipo = []

## Conjunto de dados
dados = pd.read_csv('c:\datasets\image_segmentation_2.csv', sep=';')

X = dados.iloc[:, 1:].values # visão completa
#X = dados.iloc[:, 1:7].values # visão shape
#X = dados.iloc[:, 7:].values # visão RGB
y = dados.iloc[:, 0:1].values # rótulos dos dados

c = 7 # número de clusters

# executar o algoritmo 100x
for execucoes in range(100):
    # inicialização do algoritmo
    gama = code.calcular_gama(X)
    hiperparametros = code.iniciar_hyperparametros(X, gama)
    prototipos = code.iniciar_prototipos(c, X)
    clusters = code.iniciar_afetacao_objeto(c, X, prototipos, hiperparametros)

    # parte iterativa do algoritmo
    rodadas = 0  # controlar a quantidade de iterações dentro do loop interno
    teste = 1  # variável para controlar a mudança de cluster
    while (teste != 0):
        start_total_time = time.time()
        # step 1
        prototipos = code.calcular_representantes_clusters(X, clusters, prototipos, hiperparametros)
        # step 2
        hiperparametros = code.calcular_hyperparametros(X, prototipos, clusters, gama, hiperparametros)
        # step 3
        teste = 0
        for i in range(len(X)):
            indice_atual = code.retorna_indice_atual(i, clusters)
            indice_afetado = code.atribuir_objeto_cluster(X[i], prototipos, hiperparametros)  # equação 18
            if (indice_atual != indice_afetado):
                teste += 1
                clusters[indice_atual].remove(i)
                clusters[indice_afetado].append(i)
        rodadas += 1

        # calcula a função objetivo para o cluster gerado
        valor_func_objetivo = code.funcao_objetivo(X, clusters, prototipos, hiperparametros)
        # calcula a quantidade de objetos em cada cluster
        quantidades = []
        for k in range(len(clusters)):
            quantidades.append(len(clusters[k]))
        print('Execução: ', execucoes, ' | Rodadas: ', rodadas, '| Qtd de mudanças: ', teste, '| : ', quantidades,
              ' | FuncObjetivo : ', valor_func_objetivo, ' | ', (time.time() - start_total_time), 'seconds')
    # ao sair do loop, adiciona no vetor o valor da função objetivo, cluster, hyperarametros e o protótipo
    res_func_objetivo.append(valor_func_objetivo)
    res_cluster.append(clusters)
    res_hp.append(hiperparametros)
    res_prototipo.append(prototipos)

print('##### RESULTADO FINAL #####')
indice = res_func_objetivo.index(min(res_func_objetivo))
print(indice)
print('Função Objetivo > ', res_func_objetivo[indice])
print('Clusters > ', res_cluster[indice])
print('Hiperparametros > ', res_hp[indice])
print('Protótipos > ', res_prototipo[indice])
indice_rand = code.indice_rand(X,y, res_cluster[indice])
print('Indice de Rand Ajustado > ', indice_rand)