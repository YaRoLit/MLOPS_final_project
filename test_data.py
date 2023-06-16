import pandas as pd
import pickle

from catboost import Pool

import preproc as prc
from preproc import cat_num_split


# Далее идёт блок предварительной загрузки необходимых данных:
# метаданных модели из файла метаданных,
# статистики числовых признаков из файла статистики,
# самой обученной модели,
# а также выборки, которая тестируется.
# Предусматривается, что тестируемая выборка
# всегда размещается в папке датасетов под именем train.csv

with open('model.pkl', 'rb') as file_:
    model = pickle.load(file_)

with open('model.mtd', 'r', encoding='utf-16') as file_:
    model_mtd = file_.readlines()

num_col_stats = pd.read_csv('./model_num.mtd', index_col=0, encoding='utf-16')

df = prc.load_data('./Datasets/train.csv')

df = prc.get_model_features(df)

df = prc.create_adress_feature(df)

cat_cols_uniq = {}

for cat_col in model_mtd[2:]:
    feature_name = cat_col.split(':')[0]
    feature_uniq = cat_col.split(':')[1].split(',')
    cat_cols_uniq[feature_name] = feature_uniq


def test_column_names():
    '''
    Тут мы тестируем, что в полученном датасете есть такие же
    наименования столбцов признаков, какие использовались при
    обучении модели и записаны в файле метаданных модели.
    В списке из файла метаданных убираем \n, в списке столбцов
    нашего датафрейма убираем созданный признак "адрес"
    '''
    assert model_mtd[0].split(',')[:-1].sort() == list(df.columns)[:-1].sort()


def test_unique_val_compare():
    '''
    Здесь мы смотрим, чтобы совпадали уникальные значения
    соответствующих категориальных признаков.
    '''
    for col_name in cat_cols_uniq.keys():
        assert list(df[col_name].unique()).sort() ==\
                    cat_cols_uniq[col_name][:-1].sort()


def test_build_date():
    '''
    Тут мы тестируем некоторые числовые параметры датасета
    В принципе, не сильно умный тест, но пусть будет.
    '''
    for col in num_col_stats.columns:
        # Проверяем, что в новой выборке min больше или равен, чем в старой
        assert df[col].min() >= num_col_stats[col]['min']
        # Проверяем, что в новой выборке max меньше или равен, чем в старой
        assert df[col].max() <= num_col_stats[col]['max']
        # Проверяем, что в новой выборке медиана попадает в IQR старой выборки
        assert num_col_stats[col]['25%'] <= df[col].median()\
                                         <= num_col_stats[col]['75%']


def test_df_nans_percent():
    '''Проверяем, чтобы NaN в датасете не было больше 10% в каждом столбце'''

    for col in df.columns:
        assert df[col].isna().sum() / df.shape[0] < 0.1


def test_model_acc():
    '''Проверяем точность модели на новой выборке, д.б. больше 0.74'''

    test_df = df.drop(['Столбец1', 'дом'], axis=1)

    y = test_df['стоимость м.кв.']
    X = test_df.drop('стоимость м.кв.', axis=1)
    features_names = list(test_df.drop(columns=["стоимость м.кв."]).columns)

    Data = Pool(data=X,
                label=y,
                cat_features=cat_num_split(test_df)[0],
                feature_names=features_names)

    assert model.score(Data) > 0.74
