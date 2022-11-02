# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import joblib

from starter.ml.data import process_data
from starter.ml.model import train_model
from starter.ml.model import inference
from starter.ml.model import compute_model_metrics

# Add code to load in the data.
data = pd.read_csv('./../data/census.csv')
data.columns = data.columns.str.replace(' ', '')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
x_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
x_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(x_train, y_train)
inference = inference(model, x_test)
evaluate = compute_model_metrics(y_test, inference)

filename = './../model/model.sav'
joblib.dump(model, filename)