import collections
import utils

import numpy as np
import loss_functions






def MSE(y_true: np.ndarray, y_pred: np.ndarray):
    mse = np.sum(np.square(y_true - y_pred))
    return mse


class MyLinearRegression:
    """
    Linear Regression class generalized to n-features. For description, read the method fit.

    ...

    Attributes
    ----------
    coef_ : float
        the coefficient vector

    intercept_ : float
        the intercept value

    has_intercept : bool
        whether to include intercept or not

    _fitted: bool
        a flag to turn to true once we called fit on the data

    Methods
    -------
    fit(self, X: np.ndarray = None, y_true: np.ndarray = None):
        fits the model and calculates the coef and intercept.
    """

    def __init__(
        self,
        has_intercept: bool = True,
        solver: str = "Closed Form Solution",
        learning_rate: float = 0.1,
        num_epochs: int = 1000,
    ):
        super().__init__()
        """
        Constructs all the necessary attributes for the LinearRegression object.

        Parameters
        ----------
            has_intercept : bool
                whether to include intercept or not
            
            solver: str
                {"Closed Form Solution", "Gradient Descent"}
                if Closed Form Solution: closed form solution for finding optimal parameters of beta
                                         recall \vec{beta} = (X'X)^{-1}X'Y ; note scikit-learn uses a slightly different way.
                                         https://stackoverflow.com/questions/66881829/implementation-of-linear-regression-closed-form-solution/66886954#66886954
        """

        self.solver = solver
        self.has_intercept = has_intercept
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.coef_ = None
        self.intercept_ = None
        self._fitted = False
        self.optimal_betas = None

    def _init_weights(self, X: np.ndarray):
        """
        To be included for Gradient Descent
        """
        n_features = X.shape[1]
        # init with all 0s
        initial_weights = np.zeros(shape=(n_features))  # 1d array
        return initial_weights

    def check_shape(self, X: np.ndarray, y_true: np.ndarray):
        """
        Check the shape of the inputs X & y_true

        if X is 1D array, then it is simple linear regression, reshape to 2D
        [1,2,3] -> [[1],[2],[3]] to fit the data

                Parameters:
                        X (np.ndarray): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.

                        y_true (np.ndarray): 1D numpy array (n_samples,). Input ground truth, also referred to as y_true of size m by 1.

                Returns:
                        self: Method for chaining

                Examples:
                --------
                        >>> see main

                Explanation:
                -----------

        """

        if X is not None and len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return X, y_true

    def degrees_of_freedom(self, X: np.ndarray, y_true: np.ndarray):
        # degrees of freedom of population dependent variable variance
        self._dft = self._features.shape[0] - 1
        # degrees of freedom of population error variance
        self._dfe = self._features.shape[0] - self._features.shape[1] - 1

    def fit(self, X: np.ndarray = None, y_true: np.ndarray = None):
        """
        Does not return anything. Instead it calculates the optimal beta coefficients for the Linear Regression Model. The default solver will be the closed formed solution
        B = (XtX)^{-1}Xty where we guarantee that this closed solution is unique, provided the invertibility of XtX. This is also called the Ordinary Least Squares Estimate
        where we minimze the Mean Squared Loss function to get the best beta coefficients which gives rise to the least loss function.


                Parameters:
                        X (np.ndarray): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.

                        y_true (np.ndarray): 1D numpy array (n_samples,). Input ground truth, also referred to as y_true of size m by 1.

                Returns:
                        self (MyLinearRegression): Method for chaining, as you must call .fit first on the LinearRegression class.
                                                   https://stackoverflow.com/questions/36250990/return-self-in-python

                Examples:
                --------
                        >>> see main

                Explanation:
                -----------

        """

        X, y_true = self.check_shape(X, y_true)
        n_samples, n_features = X.shape[0], X.shape[1]

        # add a column of ones if there exists an intercept: recall this is needed for intercept beta_0 whereby each sample is y_i = b1x1+b2x2+...+b0(1)
        if self.has_intercept:
            X = np.insert(X, 0, 1, axis=1)  # np.c_[np.ones(n_samples), X]

        if self.solver == "Closed Form Solution":

            XtX = np.transpose(X, axes=None) @ X
            det = np.linalg.det(XtX)
            if det == 0:
                print("Singular Matrix, Recommend to use SVD")

            XtX_inv = np.linalg.inv(XtX)
            Xty = np.transpose(X, axes=None) @ y_true
            self.optimal_betas = XtX_inv @ Xty

        elif self.solver == "Batch Gradient Descent":
            self.optimal_betas = self._init_weights(X)
            for epoch in range(self.num_epochs):
                y_pred = X @ self.optimal_betas
                MSE_LOSS = loss_functions.l2_loss(y_true, y_pred)
                GRADIENT_VECTOR = (2 / n_samples) * -(y_true - y_pred) @ X
                # yet another vectorized operation
                self.optimal_betas -= self.learning_rate * GRADIENT_VECTOR
                if epoch % 100 == 0:
                    print("EPOCH: {} | MSE_LOSS : {}".format(epoch, MSE_LOSS))

        # set attributes from None to the optimal ones
        self.coef_ = self.optimal_betas[1:]
        self.intercept_ = self.optimal_betas[0]
        self._fitted = True

        return self

    @utils.NotFitted
    def predict(self, X: np.ndarray):
        """
        Predicts the y_true value given an input of X.


                Parameters:
                        X (np.ndarray): 2D numpy array (n_samples, n_features).

                Returns:
                        y_hat: y_pred

                Examples:
                --------
                        >>> see main

                Explanation:
                -----------

        """
        if self.has_intercept:
            # y_pred = self.intercept_ + X @ self.coef_
            X = np.insert(X, 0, 1, axis=1)
            y_pred = X @ self.optimal_betas
        else:
            y_pred = X @ self.coef_

        return y_pred

    @utils.NotFitted
    def residuals(self, X: np.ndarray, y_true: np.ndarray):
        self._residuals = y_true - self.predict(X)


if __name__ == "__main__":
    """
    Notice that the intercept and coefficient values are not exactly the same when comparing sklearn's method and mine.
    This is because we are using slightly different ways to solve the question.
    Normal Equation HN: solving the normal equations by directly inverting the X.T @ X matrix.
    Normal Equation SKLEARN: On the other hand, scikit-learn uses scipy.linalg.lstsq under the hood, which uses for example an SVD-based approach.
                             That is, the mechanism there does not invert the matrix and is therefore different than yours.
                             Note that there are many ways to solve the linear least squares problem.
    """
    import sklearn
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    def regression_test():
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
        lr_HONGNAN = MyLinearRegression(solver="Closed Form Solution",
                                        has_intercept=True).fit(x_train, y_train)

        """
        Debugged, the intercept is the one with major difference why?
        """
        print(lr_SKLEARN.intercept_)
        print(lr_HONGNAN.intercept_)

        pred_HONGNAN = lr_HONGNAN.predict(x_val)
        pred_SKLEARN = lr_SKLEARN.predict(x_val)
        print("First Value HN", pred_HONGNAN[0])
        print("First Value SKLEARN", pred_SKLEARN[0])
        print("HN MSE", mean_squared_error(y_val, pred_HONGNAN))
        print("SKLEARN MSE", mean_squared_error(y_val, pred_SKLEARN))

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score

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
        lr_HONGNAN = MyLinearRegression(
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
        lr_HONGNAN = MyLinearRegression(solver="Batch Gradient Descent",
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

    # regression_diabetes()
    # regression_test()
    regression_expectation()
    # regression_diabetes()
