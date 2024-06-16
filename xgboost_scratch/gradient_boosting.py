import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from xgboost_scratch.tree import DecisionTreeRegressor
import os

class GradientBoostingRegressor:
    """
    Implements the Gradient Boosting Regressor using Friedman's method.
    """
    def __init__(self, depth=5, min_leaf=5, learning_rate=0.1, boosting_rounds=100, gamma=0, lambda_=1, alpha=0, colsample_bytree=1.0, n_jobs=1, subsample=1.0):
        self.depth = depth
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.gamma = gamma
        self.lambda_ = lambda_
        self.alpha = alpha  # L1 regularization term
        self.colsample_bytree = colsample_bytree
        self.n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
        self.subsample = subsample
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
    def negativeMeanSquaredErrorDerivative(y, y_pred):
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

        def train_boosting_tree(X_sample, pseudo_residuals_sample):
            return DecisionTreeRegressor().fit(X_sample, pseudo_residuals_sample, depth=self.depth, min_leaf=self.min_leaf, gamma=self.gamma, lambda_=self.lambda_, colsample_bytree=self.colsample_bytree)

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for booster in range(self.boosting_rounds):
                pseudo_residuals = self.negativeMeanSquaredErrorDerivative(y, self.base_pred)
                if self.subsample < 1.0:
                    sample_indices = np.random.choice(len(y), int(len(y) * self.subsample), replace=False)
                    X_sample = X[sample_indices]
                    pseudo_residuals_sample = pseudo_residuals[sample_indices]
                else:
                    X_sample = X
                    pseudo_residuals_sample = pseudo_residuals
                
                futures.append(executor.submit(train_boosting_tree, X_sample, pseudo_residuals_sample))
                
            for future in as_completed(futures):
                boosting_tree = future.result()
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
        return np.full((X.shape[0], 1), self.y_mean).flatten() + pred
