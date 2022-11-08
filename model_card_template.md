# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Develop by @zanuarts for Udacity Project 3 in Machine Learning DevOps Engineer.
- Model date: 6th November 2022.
- Model version: v2.
- Model type: classification.
- This model use Random Forest Classifier with `random_state=8`, `max_depth=64`, `n_estimator=128`.

## Intended Use

- This model can be use to predict salary of the person.
- This model can be use for decision to give someone loan.

## Training Data

- [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income).
- Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
- Prediction task is to determine whether a person makes over 50K a year.

## Evaluation Data

- Using [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income) and split it by 20% for evaluation.
- Using `fbeta`, `precision`, and `recall` to compute model metric.
- Using `OneHotEncoder` in preprocessing stage for categorical features.

## Metrics

- Using `fbeta`, `precision`, and `recall` to compute model metric.
- Model Performance on Test Data
    ```
    Precision: 0.7751430907604252
    Recall: 0.6127989657401423
    fBeta: 0.6844765342960288 
    ```

## Ethical Considerations

- The data has sensitive information of users and must be protected.

## Caveats and Recommendations

- Need to update retrain the model with other hyperparameter for the best result.
