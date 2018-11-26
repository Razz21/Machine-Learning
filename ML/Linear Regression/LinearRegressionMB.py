import numpy as np


class LinearRegressionMB(object):
    """
    Mini-Batch Gradient Descent implementation of Linear Regression model

    Parameters
    ----------

    eta : float, optional, default=0.001
        learning rate

    n_iter: int, optional, default=50
        Number of iterations (epochs)

    minibatch_size : int, optional, default=20
        Size of data samples (mini batches) used to calculate gradient in each iteration

    Attributes
    ----------

    coef_ : array, shape (n_features,)
        Weights assigned to the features.

    intercept_ : array, shape (1,)
        The intercept term.

    cost_ : array, shape (n_iter,)
        Mean cost after each iteration (epoch)

    c_  : array, shape (minibatch_size,)
        Cost calculated after exery sample
    """

    def __init__(self, n_iter=50, minibatch_size=20, eta=0.001):
        self.n_iter = n_iter
        self.minibatch_size = minibatch_size
        self.eta = eta

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
        self.cost_ = []

        self.coef_ = np.zeros(1 + X.shape[1])[1:]
        self.intercept_ = np.zeros(1 + X.shape[1])[:1]

        for epoch in range(self.n_iter):
            # shuffle data
            shuffled_indices = np.random.permutation(len(X))
            Xb = X[shuffled_indices]
            yb = y[shuffled_indices]
            self.c_ = []
            for i in range(0, len(X), self.minibatch_size):
                xi = Xb[i: i + self.minibatch_size]
                yi = yb.flatten()[i: i + self.minibatch_size]

                output = self.net_input(xi)
                errors = (yi - output)
                self.coef_ += self.eta * xi.T.dot(errors)
                self.intercept_ += self.eta * errors.sum()
                cost = (errors ** 2).sum() / 2.0
                self.c_.append(cost)
            self.cost_.append(np.mean(self.c_))

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

    def score(self, X, y, sample_weight=None):
        # copied from sklearn LinearRegression model
        """Returns the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """

        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')
