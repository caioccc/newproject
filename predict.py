import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.ensemble import AdaBoostRegressor


def split_df(df):
    train = df.sample(frac=0.8, random_state=99)
    test = df.loc[~df.index.isin(train.index), :]
    return train, test


def train_and_test(df1, real_td, predicted_td, real_ada, predicted_ada):
    # separar os dados originais em treino(80%) e test(20%) randomicamente.
    train, test = split_df(df1)
    # Com os 80% de treino, retirar a variavel alvo (price) das variaveis de predicao
    train_without_price = train[train.columns[1:7]]
    train_price = train[train.columns[0]]

    train_without_price = train_without_price.values.tolist()
    train_price = train_price.values.tolist()

    test_without_price = test[test.columns[1:7]]
    # first_line = test_without_price.iloc[[0]].values.tolist()

    clf = DecisionTreeRegressor()
    clf = clf.fit(train_without_price, train_price)

    clf_ada = AdaBoostRegressor()

    # colunas regressoras, e o alvo
    clf_ada = clf_ada.fit(train_without_price, train_price)

    # Percorrendo os dados de teste para gerar os resultados
    for i in range(0, len(test_without_price)):
        line_to_test = test_without_price.iloc[[i]].values.tolist()

        prediction_td = clf.predict(line_to_test)
        prediction_ada = clf_ada.predict(line_to_test)

        predicted_td.append(prediction_td[0])
        real_td.append(test.Price.values.tolist()[i])

        predicted_ada.append(prediction_ada[0])
        real_ada.append(test.Price.values.tolist()[i])

    return real_td, predicted_td


def main():
    df = pd.read_csv('./data/true_car_listings.csv')
    lb_make = LabelEncoder()

    # transform columns to numbers to TD
    df['city_code'] = lb_make.fit_transform(df['City'])
    df['state_code'] = lb_make.fit_transform(df['State'])
    df['make_code'] = lb_make.fit_transform(df['Make'])
    df['model_code'] = lb_make.fit_transform(df['Model'])

    # new df
    df1 = df[['Price', 'make_code', 'model_code', 'Mileage', 'Year', 'state_code', 'city_code']]
    real_td = []
    predicted_td = []
    real_ada = []
    predicted_ada = []

    for x in range(10):
        print(x)
        train_and_test(df1, real_td, predicted_td, real_ada, predicted_ada)



    list_columns = [real_td, predicted_td, real_ada, predicted_ada]
    df_to_save = pd.DataFrame(np.column_stack(list_columns),
                              columns=['real_td', 'predicted_td', 'real_ada', 'predicted_ada'])
    df_to_save.to_csv('results.csv', index=False)


    # Erro quadratico medio da raiz
    rmse_td = sqrt(mean_squared_error(real_td, predicted_td))
    rmse_ada = sqrt(mean_squared_error(real_ada, predicted_ada))

    # erro absoluto medio
    mae_td = mean_absolute_error(real_td, predicted_td)
    mae_ada = mean_absolute_error(real_ada, predicted_ada)

    # Erro percentual absoluto medio
    print('RMSE:')
    print(rmse_td)
    print(rmse_ada)
    print('MAE:')
    print(mae_td)
    print(mae_ada)


main()
