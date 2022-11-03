import sys

sys.path.append('./')

import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.json()["greeting"] == "Hello, this is Project Udacity 3!"


def test_post_prediction():
    # given
    data = json.dumps({
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    })

    # when
    r = client.post("/predict", data=data)

    # then
    assert r.json()["result"] == {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

