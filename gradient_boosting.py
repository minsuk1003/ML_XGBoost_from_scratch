## GradientBoostingRegressor Class 정의

import numpy as np
from .tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    """
    Implements the Gradient Boosting Regressor using Friedman's method.
    """
    def __init__(self, classification=False):
        self.estimators = []

    @staticmethod
    def MeanSquaredError(y, y_pred):
        """
        Computes the Mean Squared Error.
        
        Parameters
        ----------
        y : numpy array
            The true target values.
        y_pred : numpy array
            The predicted values.
        
        Returns
        -------
        float
            The mean squared error.
        """
        return np.mean((y - y_pred)**2)

    @staticmethod
    def negativeMeanSquaredErrorDerivitive(y, y_pred):
        """
        Computes the negative derivative of the Mean Squared Error.
        
        Parameters
        ----------
        y : numpy array
            The true target values.
        y_pred : numpy array
            The predicted values.
        
        Returns
        -------
        numpy array
            The negative derivative values.
        """
        return 2 * (y - y_pred)

    def fit(self, X, y, depth=5, min_leaf=5, learning_rate=0.1, boosting_rounds=5, gamma=0, lambda_=1, colsample_bytree=1.0):
        """
        Fits the gradient boosting model to the data.
        
        Parameters
        ----------
        X : pandas DataFrame
            The input features.
        y : pandas Series
            The target values.
        depth : int, optional (default=5)
            The maximum depth of each tree.
        min_leaf : int, optional (default=5)
            The minimum number of samples required to create a leaf node.
        learning_rate : float, optional (default=0.1)
            The learning rate for boosting.
        boosting_rounds : int, optional (default=5)
            The number of boosting rounds.
        gamma : float, optional (default=0)
            The regularization parameter for tree pruning.
        lambda_ : float, optional (default=1)
            The L2 regularization term on weights.
        colsample_bytree : float, optional (default=1.0)
            The subsample ratio of columns when constructing each tree.
        """
        self.learning_rate = learning_rate
        self.base_pred = np.full((X.shape[0], 1), np.mean(y)).flatten()
        for booster in range(boosting_rounds):
            pseudo_residuals = self.negativeMeanSquaredErrorDerivitive(y, self.base_pred)
            boosting_tree = DecisionTreeRegressor().fit(X, pseudo_residuals, depth=depth, min_leaf=min_leaf, gamma=gamma, lambda_=lambda_, colsample_bytree=colsample_bytree)
            self.base_pred += self.learning_rate * boosting_tree.predict(X)
            self.estimators.append(boosting_tree)

    def predict(self, X):
        """
        Predicts the target values for the input data.
        
        Parameters
        ----------
        X : pandas DataFrame
            The input features.
        
        Returns
        -------
        numpy array
            The predicted values.
        """
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)
        return np.full((X.shape[0], 1), np.mean(y)).flatten() + pred
