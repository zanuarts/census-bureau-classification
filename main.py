# Put the code for your API here.
import joblib
import numpy as np
from fastapi import FastAPI
from sklearn.preprocessing import OneHotEncoder

from starter.ml.data import process_data
from pydantic import BaseModel

# Instantiate the app.
app = FastAPI()
model_path = './model/model.sav'


class Value(BaseModel):
    age: int
    workclass: object
    fnlgt: int
    education: object
    education_num: int
    marital_status: object
    occupation: object
    relationship: object
    race: object
    sex: object
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: object

    class Config:
        schema_extra = {
            "example": {
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
        }


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello, this is Project Udacity 3!"}


@app.post("/predict")
async def predict(body: Value):
    return {"result": body}


def preprocessing(item):
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

    item_categorical = item[categorical_features].values
    item_continuous = item.drop(*[categorical_features], axis=1)
    item_categorical = encoder.transform(item_categorical)
    data = np.concatenate([item_continuous, item_categorical], axis=1)

    return data
