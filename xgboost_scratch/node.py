## Node Class 정의

import numpy as np
import random

class Node:
    """
    This class defines a node used to build a regression tree for gradient boosting.
    It uses an exact greedy method to scan every possible split point.
    
    Parameters
    ----------
    x : pandas DataFrame
        The input features.
    y : pandas Series
        The target values.
    idxs : numpy array
        The indices of the current data points.
    min_leaf : int, optional (default=5)
        The minimum number of samples required to create a leaf node.
    depth : int, optional (default=10)
        The maximum depth of the tree.
    gamma : float, optional (default=0)
        The regularization parameter for tree pruning.
    lambda_ : float, optional (default=1)
        The L2 regularization term on weights.
    colsample_bytree : float, optional (default=1.0)
        The subsample ratio of columns when constructing each tree.
    """
    def __init__(self, x, y, idxs, min_leaf=5, depth=10, gamma=0, lambda_=1, alpha=0, colsample_bytree=1.0):
        self.x, self.y = x, y
        self.idxs = idxs
        self.depth = depth
        self.min_leaf = min_leaf
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.val = self.compute_gamma(y[self.idxs])
        self.score = float('-inf')
        self.gamma = gamma
        self.lambda_ = lambda_
        self.alpha = alpha
        self.colsample_bytree = colsample_bytree
        self.selected_cols = random.sample(range(self.col_count), int(self.col_count * self.colsample_bytree))
        self.find_varsplit()

    def find_varsplit(self):
        """
        Finds the best variable and split point to perform the split.
        If no split is found that improves the score, the node is marked as a leaf.
        """
        for c in self.selected_cols: self.find_greedy_split(c)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(self.x, self.y, self.idxs[lhs], self.min_leaf, depth=self.depth-1, gamma=self.gamma, lambda_=self.lambda_, colsample_bytree=self.colsample_bytree)
        self.rhs = Node(self.x, self.y, self.idxs[rhs], self.min_leaf, depth=self.depth-1, gamma=self.gamma, lambda_=self.lambda_, colsample_bytree=self.colsample_bytree)

    def find_greedy_split(self, var_idx):
        """
        Finds the best split for a given feature by calculating the gain at each split point.
        
        Parameters
        ----------
        var_idx : int
            The index of the feature to split on.
        """
        x = self.x[self.idxs, var_idx]
        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]
            if(rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf): continue
            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]

    def gain(self, lhs, rhs):
        """
        Computes the gain for a specific split point based on Friedman's Gradient Boosting Machines.
        
        Parameters
        ----------
        lhs : numpy array
            Boolean array indicating the left split.
        rhs : numpy array
            Boolean array indicating the right split.
        
        Returns
        -------
        float
            The gain of the split.
        """
        gradient = self.y[self.idxs]
        lhs_gradient = gradient[lhs].sum()
        lhs_n_intances = len(gradient[lhs])
        rhs_gradient = gradient[rhs].sum()
        rhs_n_intances = len(gradient[rhs])
        gain = ((lhs_gradient**2 / (lhs_n_intances + self.lambda_)) + 
                (rhs_gradient**2 / (rhs_n_intances + self.lambda_)) - 
                ((lhs_gradient + rhs_gradient)**2 / (lhs_n_intances + rhs_n_intances + self.lambda_)) - self.gamma)
        return gain

    @staticmethod
    def compute_gamma(gradient):
        """
        Computes the optimal leaf node value for gradient boosting.
        
        Parameters
        ----------
        gradient : numpy array
            The gradients of the target values.
        
        Returns
        -------
        float
            The computed gamma value.
        """
        return np.sum(gradient) / len(gradient)

    @property
    def split_col(self):
        """ Returns the column values of the feature to split on. """
        return self.x[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        """ Checks if the node is a leaf node. """
        return self.score == float('-inf') or self.depth <= 0

    def predict(self, x):
        """
        Predicts the target value for the input data.
        
        Parameters
        ----------
        x : numpy array
            The input data.
        
        Returns
        -------
        numpy array
            The predicted values.
        """
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        """
        Predicts the target value for a single data point.
        
        Parameters
        ----------
        xi : numpy array
            The input data point.
        
        Returns
        -------
        float
            The predicted value.
        """
        if self.is_leaf: return self.val
        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)