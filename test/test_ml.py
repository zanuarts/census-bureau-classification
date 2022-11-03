import sys

sys.path.append("./")

import pandas as pd
import numpy as np
import pytest
import sklearn
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics


@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture
def data():
    return pd.read_csv('./data/census.csv')


def test_process_data_train(data, cat_features):
    # given
    data.columns = data.columns.str.replace(' ', '')
    train, test = train_test_split(data, test_size=0.20)

    # when
    x_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # then
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder)
    assert isinstance(lb, sklearn.preprocessing._label.LabelBinarizer)


def test_process_data_test(data, cat_features):
    # given
    data.columns = data.columns.str.replace(' ', '')
    train, test = train_test_split(data, test_size=0.20)
    x_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # when
    x_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    # then
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder)
    assert isinstance(lb, sklearn.preprocessing._label.LabelBinarizer)


def test_train_model(data, cat_features):
    # given
    data.columns = data.columns.str.replace(' ', '')
    train, test = train_test_split(data, test_size=0.20)
    x_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # when
    model = train_model(x_train, y_train)

    # then
    assert model is not None


def test_inference(data, cat_features):
    # given
    data.columns = data.columns.str.replace(' ', '')
    train, test = train_test_split(data, test_size=0.20)
    x_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    x_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    model = train_model(x_train, y_train)

    # when
    result = inference(model, x_test)

    # then
    assert isinstance(result, np.ndarray)


def test_compute_model_metrics(data, cat_features):
    # given
    data.columns = data.columns.str.replace(' ', '')
    train, test = train_test_split(data, test_size=0.20)
    x_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    x_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    model = train_model(x_train, y_train)
    result = inference(model, x_test)

    # when
    precision, recall, fbeta = compute_model_metrics(y_test, result)

    # then
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
