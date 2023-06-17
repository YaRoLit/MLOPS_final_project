import os
import pickle
import pytest

import preproc as prc
from catboost import CatBoostRegressor, Pool
from preproc import cat_num_split


@pytest.fixture(scope="module")
def model():
    df = prc.load_data('./Datasets/train.csv')
    df = prc.get_model_features(df)
    df = prc.fill_nans(df)
    df = prc.create_adress_feature(df)

    train = df.drop(['Столбец1', 'дом'], axis=1)

    X = train.drop(columns=["стоимость м.кв."])
    y = train["стоимость м.кв."]
    features_names = list(train.drop(columns=["стоимость м.кв."]).columns)

    train_data = Pool(
        data=X,
        label=y,
        cat_features=cat_num_split(train)[0],
        feature_names=features_names
    )

    model = CatBoostRegressor(
        iterations=5000,
        depth=6
    )

    model.fit(train_data)

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    yield model

    # Cleanup
    model = None
    os.remove('model.pkl')


def test_model_predict(model):
    assert model.predict([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ]) is not None
