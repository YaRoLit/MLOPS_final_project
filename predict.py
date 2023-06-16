import pickle

import pandas as pd

from catboost import CatBoostRegressor
from catboost import Pool

import preproc as prc
from preproc import cat_num_split


with open('model.pkl', 'rb') as file_:
    model = pickle.load(file_)


def make_pool(df: pd.DataFrame) -> Pool:
    '''Преобразуем данные в специальный формат Catboost'''

    return pred_data


def predict(df: pd.DataFrame) -> float:
    '''Определяем цену c помощью модели'''

    pred_data = Pool(data=df,
                cat_features=cat_num_split(df)[0],
                feature_names=list(df.columns))

    return model.predict(pred_data)


if __name__ == "__main__":

    df = prc.load_data('./Datasets/test.csv')

    df = prc.get_predict_model_features(df)

    df = prc.create_adress_feature(df)

    pred = df.drop(['Столбец1', 'дом'], axis=1)

    y_pred = predict(pred)

    pred.insert(0, 'предполагаемая цена кв.м.', y_pred)

    pred.to_csv('./Datasets/pred.csv', sep='\t', encoding='utf-16')
