import numpy as np
from .tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    """
    Implements the Gradient Boosting Regressor using Friedman's method.
    """
    def __init__(self, depth=5, min_leaf=5, learning_rate=0.1, boosting_rounds=100, gamma=0, lambda_=1, colsample_bytree=1.0, n_jobs=1):
        self.depth = depth
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.gamma = gamma
        self.lambda_ = lambda_
        self.colsample_bytree = colsample_bytree
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

    def fit(self, X, y):
        """
        Fits the gradient boosting model to the data.
        
        Parameters
        ----------
        X : pandas DataFrame
            The input features.
        y : pandas Series
            The target values.
        """
        
        self.base_pred = np.full((X.shape[0], 1), np.mean(y)).flatten()
        self.y_mean = np.mean(y)
        self.estimators = []

        for booster in range(self.boosting_rounds):
            pseudo_residuals = self.negativeMeanSquaredErrorDerivitive(y, self.base_pred)
            boosting_tree = DecisionTreeRegressor().fit(X, pseudo_residuals, depth=self.depth, min_leaf=self.min_leaf, gamma=self.gamma, lambda_=self.lambda_, colsample_bytree=self.colsample_bytree)
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
        X = X.values
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)
        return np.full((X.shape[0], 1), self.y_mean).flatten() + pred
