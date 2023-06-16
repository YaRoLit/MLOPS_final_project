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


df = prc.load_data('./Datasets/train.csv')

df = prc.get_model_features(df)

df = prc.create_adress_feature(df)

train = df.drop(['Столбец1', 'дом'], axis=1)

X = train.drop(columns = ["стоимость м.кв."])
y = train["стоимость м.кв."]
features_names = list(train.drop(columns = ["стоимость м.кв."]).columns)

train_data = Pool(data=X,
                  label=y,
                  cat_features=cat_num_split(train)[0],
                  feature_names=features_names)

model = CatBoostRegressor(iterations = 5000,
                          depth = 6)

model.fit(train_data)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)