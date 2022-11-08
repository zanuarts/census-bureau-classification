# Script to train machine learning model.
import os.path

import joblib
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np

from starter.ml.data import process_data
from starter.ml.model import train_model
from starter.ml.model import inference
from starter.ml.model import compute_model_metrics

model_path = './../model/model.pkl'
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
slice_output_path = '../slice_output.txt'

def write_output(education, precision, recall, fbeta):
    if os.path.exists(slice_output_path) is True:
        mode = 'a'
    else:
        mode = 'w'

    f = open(slice_output_path, mode)
    f.write('\nCategory: ' + education)
    f.write('\n-----')
    f.write('\nPrecision: ' + str(precision))
    f.write('\nRecall: ' + str(recall))
    f.write('\nfBeta: ' + str(fbeta))
    f.write('-----\n')
    f.close()


def education_category_performance(data, model, encoder, lb):
    unique_edu = data['education'].unique()

    for education in unique_edu:
        edu_performance = data[(data['education'] == education)]
        label_data = edu_performance['salary']
        edu_performance = edu_performance.drop(['salary'], axis=1)

        try:
            label = lb.transform(label_data.values).ravel()
        except AttributeError:
            print("There is no such attribute")

        item_categorical = edu_performance[cat_features].values
        item_continuous = edu_performance.drop(*[cat_features], axis=1)
        item_categorical = encoder.transform(item_categorical)

        prediction_data = np.concatenate([item_continuous, item_categorical], axis=1)
        result = inference(model, prediction_data)
        precision, recall, fbeta = compute_model_metrics(label, result)

        write_output(education, precision, recall, fbeta)


def load_data():
    # Add code to load in the data.
    data = pd.read_csv('./../data/census.csv')
    data.columns = data.columns.str.replace(' ', '')
    data.columns = data.columns.str.replace('-', '_')

    return data


def preprocessing(data):
    # Optional enhancement, use K-fold cross validation
    # instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    x_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # Proces the test data with the process_data function.
    x_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "encoder": encoder,
        "lb": lb,
    }


def main():
    data = load_data()
    data_preprocessed = preprocessing(data)

    x_train = data_preprocessed['x_train']
    y_train = data_preprocessed['y_train']
    x_test = data_preprocessed['x_test']
    y_test = data_preprocessed['y_test']
    encoder = data_preprocessed['encoder']
    lb = data_preprocessed['lb']

    model = train(x_train, y_train, x_test, y_test, encoder, lb)
    education_category_performance(data, model, encoder, lb)


def train(x_train, y_train, x_test, y_test, encoder, lb):
    # Train and save a model.
    model = train_model(x_train, y_train)
    result = inference(model, x_test)
    precision, recall, fbeta = compute_model_metrics(y_test, result)

    f = open(slice_output_path, 'w')
    f.write('\nModel Performance on Test Data')
    f.write('\n-----')
    f.write('\nPrecision: ' + str(precision))
    f.write('\nRecall: ' + str(recall))
    f.write('\nfBeta: ' + str(fbeta))
    f.write('\n-----\n')
    f.close()

    joblib.dump((model, encoder, lb), model_path)

    return model


if __name__ == '__main__':
    main()
