import pickle

import pandas as pd

from catboost import CatBoostRegressor
from catboost import Pool

import preproc as prc
from preproc import cat_num_split


with open('model.pkl', 'rb') as file_:
    model = pickle.load(file_)


def predict(df: pd.DataFrame) -> float:
    '''Определяем цену c помощью модели'''

    pred_data = Pool(data=df,
                cat_features=cat_num_split(df)[0],
                feature_names=list(df.columns))

    return model.predict(pred_data)


def predict_test_file() -> None:
    '''
    Берем файл test.csv из папки с /Datasets
    и добавляем ему колонку с ценой кв.м. 
    '''

    df = prc.load_data('./Datasets/test.csv')

    df = prc.get_predict_model_features(df)

    df = prc.create_adress_feature(df)

    pred = df.drop(['Столбец1', 'дом'], axis=1)

    y_pred = predict(pred)

    pred.insert(0, 'предполагаемая цена кв.м.', y_pred)

    pred.to_csv('./Datasets/pred.csv', sep='\t', encoding='utf-16')    
    

if __name__ == "__main__":
   try:
       predict_test_file()

   except:
       print('В папке /Datasets нет файла test.csv или он косячный')
