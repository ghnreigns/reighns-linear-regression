from __future__ import print_function

from torch._C import device

from argparse import Namespace
import mlflow.pytorch
import mlflow
# Trains using PyTorch and logs training metrics and weights in TensorFlow event format to the MLflow run's artifact directory.
# This stores the TensorFlow events in MLflow for later access using TensorBoard.
#
# Code based on https://github.com/mlflow/mlflow/blob/master/example/tutorial/pytorch_tensorboard.py.
#


import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from typing import *
from pathlib import Path
from chardet.universaldetector import UniversalDetector
from src import linear_regression, loss_functions
import typer
import torchvision
import sklearn
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Typer CLI app
app = typer.Typer()


"""
Notice that the intercept and coefficient values are not exactly the same when comparing sklearn's method and mine.
This is because we are using slightly different ways to solve the question.
Normal Equation HN: solving the normal equations by directly inverting the X.T @ X matrix.
Normal Equation SKLEARN: On the other hand, scikit-learn uses scipy.linalg.lstsq under the hood, which uses for example an SVD-based approach.
                            That is, the mechanism there does not invert the matrix and is therefore different than yours.
                            Note that there are many ways to solve the linear least squares problem.
"""

@app.command()
def regression_test(solver:str="Closed Form Solution", num_epochs: int=None):
    np.random.seed(1930)
    X, y_true = make_regression(
        n_samples=10000, n_features=10, random_state=1930, coef=False
    )  # returns 2-d array of 1000 by 10
    x_train, x_val, y_train, y_val = train_test_split(
        X, y_true, test_size=0.3, random_state=1930
    )

    lr_SKLEARN = LinearRegression(fit_intercept=True, normalize=False).fit(
        x_train, y_train
    )
    lr_HONGNAN = linear_regression.MyLinearRegression(solver=solver,
                                    has_intercept=True, num_epochs=num_epochs).fit(x_train, y_train)

    """
    Debugged, the intercept is the one with major difference why? Answer: https://stackoverflow.com/questions/66881829/implementation-of-linear-regression-closed-form-solution/66886954?noredirect=1#comment118259946_66886954
    """
    print(lr_SKLEARN.intercept_)
    print(lr_HONGNAN.intercept_)

    pred_HONGNAN = lr_HONGNAN.predict(x_val)
    pred_SKLEARN = lr_SKLEARN.predict(x_val)
    print("First Value HN", pred_HONGNAN[0])
    print("First Value SKLEARN", pred_SKLEARN[0])
    print("HN MSE", mean_squared_error(y_val, pred_HONGNAN))
    print("SKLEARN MSE", mean_squared_error(y_val, pred_SKLEARN))


@app.command()
def regression_diabetes():
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    seen = set()

    uniq = [x for x in diabetes_X_train.flatten() if x in seen or seen.add(x)]

    l = [ll for ll in diabetes_X_train.flatten() if ll == 0.0616962065186885]
    print(l)
    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    lr_HONGNAN = linear_regression.MyLinearRegression(
        has_intercept=True, solver="Batch Gradient Descent", num_epochs=6666
    ).fit(diabetes_X_train, diabetes_y_train)

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)
    pred_HONGNAN = lr_HONGNAN.predict(diabetes_X_test)

    # The coefficients
    print("SKLEARN Coefficients:", regr.coef_)
    print("HONGNAN Coefficients:", lr_HONGNAN.coef_)
    # The mean squared error
    print("HN MSE", mean_squared_error(diabetes_y_test, pred_HONGNAN))
    print("SKLEARN MSE", mean_squared_error(
        diabetes_y_test, diabetes_y_pred))

    # The coefficient of determination: 1 is perfect prediction
    print(
        "Coefficient of determination: %.2f"
        % r2_score(diabetes_y_test, diabetes_y_pred)
    )

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

def regression_expectation():
    np.random.seed(1930)
    # # Load the diabetes dataset
    # diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # # Use only one feature
    # diabetes_X = diabetes_X[:, np.newaxis, 2]

    # # Split the data into training/testing sets
    # diabetes_X_train = diabetes_X[:-20]
    # diabetes_X_test = diabetes_X[-20:]
    # seen = set()

    # uniq = [x for x in diabetes_X_train.flatten() if x in seen or seen.add(x)]

    # uniq_index = np.where(diabetes_X_train.flatten() == 0.0616962065186885)
    # print(uniq_index)

    # # Split the targets into training/testing sets
    # diabetes_y_train = diabetes_y[:-20]
    # diabetes_y_test = diabetes_y[-20:]

    # uniq_y_expectation = np.mean(diabetes_y_train[uniq_index])
    # print(diabetes_y_train[uniq_index])  # 196
    # # Create linear regression object
    # regr = linear_model.LinearRegression()
    # lr_HONGNAN = MyLinearRegression(
    #     has_intercept=True, solver="Closed Form Solution",
    # ).fit(diabetes_X_train, diabetes_y_train)

    # # Train the model using the training sets
    # regr.fit(diabetes_X_train, diabetes_y_train)

    # # Make predictions using the testing set
    # diabetes_y_pred = regr.predict(diabetes_X_test)
    # pred_HONGNAN = lr_HONGNAN.predict(diabetes_X_test)

    # # The coefficients
    # print("SKLEARN Coefficients:", regr.coef_)
    # print("HONGNAN Coefficients:", lr_HONGNAN.coef_)
    # print(lr_HONGNAN.predict([[0.0616962065186885]]))

    X, y_true, true_param = make_regression(
        n_samples=10000, n_features=1, random_state=1930, coef=True, shuffle=False, tail_strength=0, noise=0.0, bias=0.01
    )  # returns 2-d array of 1000 by 10
    x_train, x_val, y_train, y_val = train_test_split(
        X, y_true, test_size=0.3, random_state=1930
    )

    print(true_param)

    # Purposely assign some repeated values

    x_train[0:1000] = 0.5

    # ensure 0.5 only exists in x_train[0:10] and no where else

    y_corresponding = y_train[0:1000]
    print(np.mean(y_train), np.mean(x_train))
    expectation_y_corresponding = np.mean(y_corresponding)
    print(expectation_y_corresponding)

    seen = set()

    uniq = [x for x in x_train.flatten() if x in seen or seen.add(x)]
    # print(uniq)

    x_train = sklearn.preprocessing.scale(x_train)

    lr_SKLEARN = LinearRegression(fit_intercept=True, normalize=False).fit(
        x_train, y_train
    )
    # lr_HONGNAN = MyLinearRegression(solver="Closed Form Solution",
    #                                 has_intercept=True).fit(x_train, y_train)
    lr_HONGNAN = linear_regression.MyLinearRegression(solver="Batch Gradient Descent",
                                    has_intercept=True).fit(x_train, y_train)
    """
    Debugged, the intercept is the one with major difference why?
    """
    print(lr_SKLEARN.coef_)
    print(lr_HONGNAN.coef_)
    print(lr_SKLEARN.intercept_)

    pred_HONGNAN = lr_HONGNAN.predict(x_val)
    pred_SKLEARN = lr_SKLEARN.predict(x_val)
    print(lr_HONGNAN.predict([[0.5]]))
    # print("First Value HN", pred_HONGNAN[0])
    # print("First Value SKLEARN", pred_SKLEARN[0])
    # print("HN MSE", mean_squared_error(y_val, pred_HONGNAN))
    # print("SKLEARN MSE", mean_squared_error(y_val, pred_SKLEARN))

