import requests

from main import preprocessing

endpoint = 'https://obscure-caverns-94591.herokuapp.com/predict'
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
    "native_country": "United-States"
}


def main():
    response = requests.post(endpoint, json=data)

    print(response.status_code)
    print(response.json())


if __name__ == '__main__':
    main()
