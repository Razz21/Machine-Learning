# Original Source:
#   Python Machine Learning by Sebastian Raschka


import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
#BaseEstimator class add get_params/set_params method for estimators in sckit-learn
#RegressionMixin class add score method for regression estimators in scikit-learn

class LinearRegressionGD(BaseEstimator,RegressorMixin):
    """
    Batch Gradient Descent implementation of Linear Regression model

    Parameters
    ----------

    eta : float, optional, default=0.001
        learning rate

    n_iter: int, optional, default=20
        Number of iterations (epochs)

    Attributes
    ----------

    coef_ : array, shape (n_features,)
        Weights assigned to the features.

    intercept_ : array, shape (1,)
        The intercept term.

    cost_ : array, shape (n_iter,)
        Cost calculated after each iteration (epoch)

    """

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data

        y : array_like, shape (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """

        self.coef_ = np.zeros(1 + X.shape[1])[1:]
        self.intercept_ = np.zeros(1 + X.shape[1])[:1]
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.coef_ += self.eta * X.T.dot(errors)
            self.intercept_ += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Predict values using the linear model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples,)
            Predicted target values per element in X.
        """

        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape (n_samples,)
            Returns predicted values.
        """
        return self.net_input(X)
 
