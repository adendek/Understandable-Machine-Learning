import numpy as np
#from src.SupervisedLearning.LinearRegression.Validator import LinRegValidator
#from src.SupervisedLearning.LinearRegression.utils import *


class LinearRegressionModel:
    def __init__(self, n_features, weights=None, optimizer="GD", measure="MSE"):
        self.n_features = n_features
        if weights:
            self.weights = weights
        else:
            self.weights = np.random.rand(self.n_features, 1)
        self.optimizer = optimizer
        self.measure = measure
        #self.validator = LinRegValidator(n_features=n_features)
        
    def fit(self, data, target):
        #self.validator.validate_training(data, target)
        self.weights = self.optimizer.optimize(data, target, self.measure)
        
    def predict(self, data):
        data = np.c_[np.ones((data.shape[0], 1)), data]
        return np.dot(data, self.weights)
        
    
