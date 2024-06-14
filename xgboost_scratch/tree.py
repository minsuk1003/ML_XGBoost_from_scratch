## DecisionTreeRegressor 정의
import numpy as np
from .node import Node

class DecisionTreeRegressor:
    """
    Wrapper class for the regression tree that provides a scikit-learn interface.
    """
    def fit(self, X, y, min_leaf=5, depth=5, gamma=0, lambda_=1, colsample_bytree=1.0):
        """
        Fits the regression tree to the data.
        
        Parameters
        ----------
        X : pandas DataFrame
            The input features.
        y : pandas Series
            The target values.
        min_leaf : int, optional (default=5)
            The minimum number of samples required to create a leaf node.
        depth : int, optional (default=5)
            The maximum depth of the tree.
        gamma : float, optional (default=0)
            The regularization parameter for tree pruning.
        lambda_ : float, optional (default=1)
            The L2 regularization term on weights.
        colsample_bytree : float, optional (default=1.0)
            The subsample ratio of columns when constructing each tree.
        
        Returns
        -------
        DecisionTreeRegressor
            The fitted regressor.
        """
        self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf, depth, gamma, lambda_, colsample_bytree)
        return self

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
        return self.dtree.predict(X)
