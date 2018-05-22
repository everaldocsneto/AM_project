from sklearn.model_selection import StratifiedKFold
import pandas as pd
import classificadores_final.bayesiano_gauss as gauss
import classificadores_final.bayesiano_parzen as parzen
import classificadores_final.regra_soma as soma
import classificadores_final.metricas as metricas

dados = pd.read_csv('c:\datasets\image_segmentation.csv', sep=';')
X = dados.iloc[:, 1:].values
y = dados.iloc[:, 0:1].values

# gerar o resultado em arquivo
arquivo = open('C:/datasets/resultado_AM.txt','w')
arquivo.write('RODADA|COMPLETE (GAUSS)|SHAPE (GAUSS)|RGB (GAUSS)|COMPLETE (PARZEN)|SHAPE (PARZEN)|RGB (PARZEN)|REGRA SOMA(Completo)|SOMA (V1)|SOMA (V2)|SOMA (V3)' + '\n')

for j in range(30):  # repetir 30x
    skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)  # 10 rodadas
    contador = 1
    gv1, gv2, gv3 = 0, 0, 0
    pv1, pv2, pv3 = 0, 0, 0
    sv1, sv2, sv3 = 0, 0, 0
    rs = 0

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # ajuste dos dados da visão
        complete_view_train = X_train
        shape_view_train = X_train[:, :9]
        rgb_view_train = X_train[:, 9:]

        complete_view_test = X_test  # visão 1
        shape_view_test = X_test[:, :9]  # visão 2
        rgb_view_test = X_test[:, 9:]  # visão 3

        parametros = gauss.estimar_parametros(complete_view_train, y_train)
        classes = parametros[0]
        priori = parametros[3]

        resultado_gauss_v1, resultado_gauss_v2, resultado_gauss_v3 = [], [], []
        resultado_parzen_v1, resultado_parzen_v2, resultado_parzen_v3 = [], [], []
        resultado_soma = []
        # acrescentar
        resultado_soma_v1, resultado_soma_v2, resultado_soma_v3 = [], [], []

        # lista de probabilidades a priori de cada classificador
        posteriori_gauss_1 = gauss.executar_gauss(complete_view_train, y_train, complete_view_test)
        posteriori_gauss_2 = gauss.executar_gauss(shape_view_train, y_train, shape_view_test)
        posteriori_gauss_3 = gauss.executar_gauss(rgb_view_train, y_train, rgb_view_test)
        posteriori_parzen_1 = parzen.executar_parzen(complete_view_train, y_train, complete_view_test)
        posteriori_parzen_2 = parzen.executar_parzen(shape_view_train, y_train, shape_view_test)
        posteriori_parzen_3 = parzen.executar_parzen(rgb_view_train, y_train, rgb_view_test)
        posteriori_soma = soma.calcular_regra_soma(posteriori_gauss_1, posteriori_gauss_2, posteriori_gauss_3,
                                                   posteriori_parzen_1, posteriori_parzen_2, posteriori_parzen_3,
                                                   priori)

        posteriori_soma_v1 = soma.calcular_regra_soma_visao(posteriori_gauss_1, posteriori_parzen_1)
        posteriori_soma_v2 = soma.calcular_regra_soma_visao(posteriori_gauss_2, posteriori_parzen_2)
        posteriori_soma_v3 = soma.calcular_regra_soma_visao(posteriori_gauss_3, posteriori_parzen_3)

        # para cada lista de probabilidades, escolher a maior e associar a classe
        for i in range(len(posteriori_gauss_1)):
            resultado_gauss_v1.append(metricas.escolha_classe(posteriori_gauss_1[i], classes))
            resultado_gauss_v2.append(metricas.escolha_classe(posteriori_gauss_2[i], classes))
            resultado_gauss_v3.append(metricas.escolha_classe(posteriori_gauss_3[i], classes))
            resultado_parzen_v1.append(metricas.escolha_classe(posteriori_parzen_1[i], classes))
            resultado_parzen_v2.append(metricas.escolha_classe(posteriori_parzen_2[i], classes))
            resultado_parzen_v3.append(metricas.escolha_classe(posteriori_parzen_3[i], classes))
            resultado_soma.append(metricas.escolha_classe(posteriori_soma[i], classes))
            # acrescentando
            resultado_soma_v1.append(metricas.escolha_classe(posteriori_soma_v1[i], classes))
            resultado_soma_v2.append(metricas.escolha_classe(posteriori_soma_v2[i], classes))
            resultado_soma_v3.append(metricas.escolha_classe(posteriori_soma_v3[i], classes))

        # calcular a quantidade de acertos
        gv1 = gv1 + metricas.qtd_acertos(resultado_gauss_v1, y_test)
        gv2 = gv2 + metricas.qtd_acertos(resultado_gauss_v2, y_test)
        gv3 = gv3 + metricas.qtd_acertos(resultado_gauss_v3, y_test)
        pv1 = pv1 + metricas.qtd_acertos(resultado_parzen_v1, y_test)
        pv2 = pv2 + metricas.qtd_acertos(resultado_parzen_v2, y_test)
        pv3 = pv3 + metricas.qtd_acertos(resultado_parzen_v3, y_test)
        rs = rs + metricas.qtd_acertos(resultado_soma, y_test)
        #acrescentando
        sv1 = sv1 + metricas.qtd_acertos(resultado_soma_v1, y_test)
        sv2 = sv2 + metricas.qtd_acertos(resultado_soma_v2, y_test)
        sv3 = sv3 + metricas.qtd_acertos(resultado_soma_v3, y_test)


        contador +=1

    print('####### RESULTADO #######')
    print('Complete View (Gauss) - Rodada ', j, ' - ', gv1,  ' taxa de acerto >> ', gv1/2100)
    print('Shape View (Gauss) - Rodada ', j, ' - ', gv2,  ' taxa de acerto >> ', gv2 / 2100)
    print('RGB View (Gauss) - Rodada ', j, ' - ', gv3,  ' taxa de acerto >> ', gv3 / 2100)
    print('Complete View (Parzen) - Rodada ', j, ' - ', pv1, ' taxa de acerto >> ', pv1 / 2100)
    print('Shape View (Parzen) - Rodada ', j, ' - ', pv2, ' taxa de acerto >> ', pv2 / 2100)
    print('RGB View (Parzen) - Rodada ', j, ' - ', pv3, ' taxa de acerto >> ', pv3 / 2100)
    print('Regra da Soma (all) - Rodada ', j, ' - ', rs, ' taxa de acerto >> ', rs / 2100)
    # acrescentando
    print('Regra da Soma (visão 1) - Rodada ', j, ' - ', sv1, ' taxa de acerto >> ', sv1 / 2100)
    print('Regra da Soma (visão 2) - Rodada ', j, ' - ', sv2, ' taxa de acerto >> ', sv2 / 2100)
    print('Regra da Soma (visão 3) - Rodada ', j, ' - ', sv3, ' taxa de acerto >> ', sv3 / 2100)

    arquivo.write(str(j) + '|' + str(gv1/2100) + '|' + str(gv2/2100) + '|' + str(gv3/2100) + '|' + str(pv1/2100) + '|' + str(pv2/2100) + '|'
    + str(pv3 / 2100) + '|' + str(rs/2100) + '|' + str(sv1/2100) + '|' + str(sv2/2100) + '|' + str(sv3/2100) + '\n')