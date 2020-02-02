# Summary

This package contains a ridge regression algorithm written from scratch and trained/tested on a dataset to predict the median house value given a list of predictors. To check the correctness of the implemented algorithm, scikit-learn's `Ridge` regression estimator is also trained on the same training set and tested on the same test set. Its resulting performance is compared with that of the custom built ridge regression algorithm.

# Description

The primary objective of this project was to accurately translate the mathematics behind the ridge regression method and batch gradient descent into code.

Hence, the focus here is *NOT* to maximise the prediction accuracy or visualize the data and perform data exploration.

The categorical feature 'RAD' has been one-hot encoded and the other continuous features have been standardized to zero mean and unit variance to have comparable scales.

Finally, to test whether the implemented ridge regression algorithm is working as expected, its performance on the test set is compared with that of the Ridge Regression estimator found in scikit-learn.

## Ridge Regression

The objective function, regularized with the L2 norm, to be optimized is:

$||y - Xw||^2_2 + alpha*||w||^2_2$

where,

    y = vector of true values
    X = design matrix made of input features
    w = weight vector of the ridge regression model
    alpha = regularization strength parameter

The optimization algorithm of choice is a simple batch gradient descent algorithm.

## Data

The data set used is the Boston Housing data set from UCI Repository. Effectively, it contains 506 rows of data, where each row consists of several possible indicative features that could help in determining the correponding median house value.

### Predictors

- **CRIM** - per capita crime rate by town
- **ZN** - proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS** - proportion of non-retail business acres per town.
- **CHAS** - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX** - nitric oxides concentration (parts per 10 million)
- **RM** - average number of rooms per dwelling
- **AGE** - proportion of owner-occupied units built prior to 1940
- **DIS** - weighted distances to five Boston employment centres
- **RAD** - index of accessibility to radial highways
- **TAX** - full-value property-tax rate per $10,000
- **PTRATIO** - pupil-teacher ratio by town
- **B** - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- **LSTAT** - % lower status of the population

### Response

- **MEDV** - Median value of owner-occupied homes in $1000's

Ref: <https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html>

## Requirements

- Python 3.7.4
- Python dependencies:
  - [Tox](https://tox.readthedocs.io/en/latest/) (optional; needed for testing)
  - pandas
  - numpy
  - scikit-learn

## Installation

1. Clone this repo
2. Move into the cloned directory.
3. Run `pip install .`

## Usage

Assuming you've installed the package, simply run the following in the terminal:

`ridge-reg`

The following should be the resultant output:

```
Mean Absolute Percentage Error of custom model on test set: 18.04574010314292%
Mean Absolute Percentage Error of sklearn model on test set: 18.433893328746397%
```

## Testing

There are 3 types of automated testing provisioned for this package:

1. Unit/integration tests
2. Pylint
3. Flake8

All of these will require that you have `tox` installed in your system.

`pip install tox`

### Unit/Integration Tests

While in the cloned package directory, simply run the following in a terminal/cmd:

`tox`

### Pylint

While in the cloned package directory, run the following:

`tox -e pylint`

Which should give a comprehensive report, ending with:

```
--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```

### Flake8

While in the cloned package directory, run the following:

`tox -e flake8`

## Author

- Akaash Agarwal
