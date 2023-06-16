import pickle

import pandas as pd

from catboost import CatBoostRegressor
from catboost import Pool

import preproc as prc


def cat_num_split(df: pd.DataFrame) -> tuple:
    '''Ищем категориальные и числовые признаки в датафрейме'''

    cat_columns = []
    num_columns = []

    for column_name in df.columns:
        if df[column_name].dtypes == object:
            cat_columns += [column_name]
        else:
            num_columns += [column_name]

    return cat_columns, num_columns


df = prc.load_data('./Datasets/test.csv')

df = prc.get_predict_model_features(df)

df = prc.create_adress_feature(df)

pred = df.drop(['Столбец1', 'дом'], axis=1)

X = pred
features_names = list(pred.columns)

pred_data = Pool(data=X,
                 cat_features=cat_num_split(pred)[0],
                 feature_names=features_names)

with open('model.pkl', 'rb') as file_:
    model = pickle.load(file_)

y_pred = model.predict(pred_data)

pred.insert(0, 'предполагаемая цена кв.м.', y_pred)

pred.to_csv('./Datasets/pred.csv', sep='\t', encoding='utf-16')