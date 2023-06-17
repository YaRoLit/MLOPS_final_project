import os

import pandas as pd

from preproc import fill_nans,\
                    cat_num_split,\
                    load_data,\
                    create_adress_feature\


df_test = pd.DataFrame({'A': [1, 2, 1, 4, None, None, 7, 1, 9],
                        'B': [None, 8, 7, 6, 7, None, 7, 2, 1],
                        'Столбец1': ['a', 'b', None, 'c',
                                     'b', 'b', 'a', 'b', 'c'],
                        'дом': ['a', None, None, 'a',
                                None, 'c', None, 'a', None]})


def test_fill_nans():
    '''Тестируем функцию заполнения пропусков'''

    df_nonans = pd.DataFrame({'A': [1.0, 2.0, 1.0, 4.0, 2.0,
                                    2.0, 7.0, 1.0, 9.0],
                              'B': [7.0, 8.0, 7.0, 6.0, 7.0,
                                    7.0, 7.0, 2.0, 1.0],
                              'Столбец1': ['a', 'b', 'b', 'c',
                                           'b', 'b', 'a', 'b', 'c'],
                              'дом': ['a', 'a', 'a', 'a',
                                      'a', 'c', 'a', 'a', 'a']})

    assert fill_nans(df_test).equals(df_nonans)


def test_cat_num_split():
    '''Тестируем функцию разделения категориальных и числовых столбцов'''

    assert cat_num_split(df_test) == (['Столбец1', 'дом'], ['A', 'B'])


def test_load_data():
    '''Тестируем функцию загрузки датафреймов'''

    df_test.to_csv('tst.csv', sep='\t', encoding='utf-16', index=False)

    assert load_data('tst.csv').equals(df_test)

    os.remove('tst.csv')


def test_create_adress_feature():
    '''Тестируем функцию добавления признака "адрес"'''

    df_adress = pd.DataFrame({'A': [1.0, 2.0, 1.0, 4.0, 2.0,
                                    2.0, 7.0, 1.0, 9.0],
                              'B': [7.0, 8.0, 7.0, 6.0, 7.0,
                                    7.0, 7.0, 2.0, 1.0],
                              'Столбец1': ['a', 'b', 'b', 'c',
                                           'b', 'b', 'a', 'b', 'c'],
                              'дом': ['a', 'a', 'a', 'a', 'a',
                                      'c', 'a', 'a', 'a'],
                              'адрес': ['a, a', 'b, a', 'b, a', 'c, a',
                                        'b, a', 'b, c', 'a, a', 'b, a',
                                        'c, a']})

    assert create_adress_feature(df_test).equals(df_adress)
