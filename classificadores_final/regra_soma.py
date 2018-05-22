# recebe uma lista com as probabilidades posteriores dos classificadores guassiano e parzen para cada visão
def calcular_regra_soma(p_gauss_v1, p_gauss_v2, p_gauss_v3, p_parzen_v1, p_parzen_v2, p_parzen_v3, priori):
    posteriori_soma = []
    for i in range(len(p_gauss_v1)):
        p_soma = []
        for j in range(len(p_gauss_v1[i])):
            p_soma.append(((1-3)*priori[j])  + (p_gauss_v1[i][j]+p_gauss_v2[i][j]+p_gauss_v3[i][j]+p_parzen_v1[i][j]+p_parzen_v2[i][j]+p_parzen_v3[i][j]))
        posteriori_soma.append(p_soma)
    return posteriori_soma

def calcular_regra_soma_visao(p_gauss, p_parzen):
    posteriori_soma = []
    for i in range(len(p_gauss)):
        p_soma = []
        for j in range(len(p_gauss[i])):
            p_soma.append(p_gauss[i][j]+p_parzen[i][j])
        posteriori_soma.append(p_soma)
    return posteriori_soma