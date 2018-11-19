# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.ensemble import AdaBoostRegressor

#Split dos dados originais em 80% para treino e 20% para teste randomicamente
def split_df(df):
    train = df.sample(frac=0.8)
    test = df.loc[~df.index.isin(train.index), :]
    return train, test


def train_and_test(df1, real_td, predicted_td, real_ada, predicted_ada):
    #Separando os dados originais em treino(80%) e test(20%) randomicamente.
    train, test = split_df(df1)

    # Com os 80% de treino, retirar a variavel alvo (price) das variaveis de predicao
    train_without_price = train[train.columns[1:7]] #DF sem a coluna preço
    train_price = train[train.columns[0]] #DF só com a coluna preço

    # Transformando o DF em lista de listas, onde cada lista interna é uma linha do DF
    train_without_price = train_without_price.values.tolist()
    train_price = train_price.values.tolist()

    test_without_price = test[test.columns[1:7]] #Pegando apenas as variáveis de predição

    clf = DecisionTreeRegressor() #Criando o modelo da árvore de decisão
    clf = clf.fit(train_without_price, train_price) #Treinando a árvore de decisão, passando variaveis preditoras e variável alvo

    clf_ada = AdaBoostRegressor() #Criando o modelo ada booster
    clf_ada = clf_ada.fit(train_without_price, train_price) #Treinando o ada booster

    prediction_td = clf.predict(test_without_price)  # Prevendo na árvore de decisão
    prediction_ada = clf_ada.predict(test_without_price) #Prevendo no ada booster

    predicted_td.extend(prediction_td.tolist())
    real_td.extend(test.Price.values.tolist())

    predicted_ada.extend(prediction_ada.tolist())
    real_ada.extend(test.Price.values.tolist())

    # # Percorrendo os dados de teste para coletar o valor da predição
    # for i in range(0, len(test_without_price)):
    #     line_to_test = test_without_price.iloc[[i]].values.tolist() #Linha do CSV para realizar predição
    #
    #     prediction_td = clf.predict(line_to_test) #Prevendo na árvore de decisão
    #     prediction_ada = clf_ada.predict(line_to_test) #Prevendo no ada booster
    #
    #     #Adicionando os resultados da predição e valor real às suas respectivas listas
    #     predicted_td.append(prediction_td[0])
    #     real_td.append(test.Price.values.tolist()[i])
    #
    #     predicted_ada.append(prediction_ada[0])
    #     real_ada.append(test.Price.values.tolist()[i])
    #
    #     print((i*100)/(len(test_without_price))) #Mostrando % percorrida dos testes


def main():
    #Lendo o CSV
    df = pd.read_csv('./data/true_car_listings.csv')
    lb_make = LabelEncoder()

    #Transformando as colunas categóricas em numéricas
    df['city_code'] = lb_make.fit_transform(df['City'])
    df['state_code'] = lb_make.fit_transform(df['State'])
    df['make_code'] = lb_make.fit_transform(df['Make'])
    df['model_code'] = lb_make.fit_transform(df['Model'])

    #Criando um novo DataFrame com as novas colunas numéricas
    df1 = df[['Price', 'make_code', 'model_code', 'Mileage', 'Year', 'state_code', 'city_code']]

    #Listas dos preços reais e preditos das duas IAs. (Árvore de decisão e Ada Booster)
    real_td = []
    predicted_td = []
    real_ada = []
    predicted_ada = []

    #Refazendo o split, treinando e predizendo
    for x in range(10):
        print(x)
        train_and_test(df1, real_td, predicted_td, real_ada, predicted_ada)



    #Criando as colunas do CSV
    list_columns = [real_td, predicted_td, real_ada, predicted_ada]
    #Criando o CSV
    df_to_save = pd.DataFrame(np.column_stack(list_columns),
                              columns=['real_td', 'predicted_td', 'real_ada', 'predicted_ada'])
    #Salvando o CSV
    df_to_save.to_csv('results.csv', index=False)


    # Erro quadratico medio da raiz
    rmse_td = sqrt(mean_squared_error(real_td, predicted_td))
    rmse_ada = sqrt(mean_squared_error(real_ada, predicted_ada))

    # erro absoluto medio
    mae_td = mean_absolute_error(real_td, predicted_td)
    mae_ada = mean_absolute_error(real_ada, predicted_ada)

    print('RMSE:')
    print(rmse_td)
    print(rmse_ada)
    print('MAE:')
    print(mae_td)
    print(mae_ada)


main()
