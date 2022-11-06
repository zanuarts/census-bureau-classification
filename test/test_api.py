import sys

sys.path.append('./')

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["greeting"] == "Hello, this is Project Udacity 3!"


def test_get_malformed():
    r = client.get("/model")
    assert r.status_code != 200


def test_post_prediction():
    # given
    data = {
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

    # when
    r = client.post("/predict", json=data)

    # then
    assert r.status_code == 200
    assert r.json()["result"] == "<=50k"

def test_post_prediction_malformed():
    # given
    data = {
        "age": 39,
    }

    # when
    r = client.post("/predict", json=data)

    # then
    assert r.status_code != 200
