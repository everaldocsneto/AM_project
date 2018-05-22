def qtd_acertos(resultado, labels):
    acertos = 0
    for i in range(len(labels)):
        if (resultado[i] == labels[i]):
            acertos += 1
    return acertos

# retorna a classe com maior probabilidade posteriori
def escolha_classe(p_posteriori, classes):
    return classes[p_posteriori.index(max(p_posteriori))]