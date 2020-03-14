import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import numpy as np
from Optimizer import BatchGradientDecent
from Loss import MSE
from Validator import LinRegValidator


class LinearRegressionModel:
    def __init__(self, n_features, optimizer=None, loss=None):
        self.n_features = n_features
        self.weights = np.zeros(n_features + 1) # add weight for bias term 
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = BatchGradientDecent(learning_rate=1, n_steps=100, save_history=True)
        if loss:
            self.loss = loss
        else:
            self.loss = MSE()        
        self.validator = LinRegValidator(n_features=n_features)
        
    def fit(self, data, target):
        target = target.reshape(-1,)
        self.validator.validate_training(data, target)
        self.weights = self.optimizer.optimize(data, target, loss=self.loss, weights=self.weights)
        
    def predict(self, data):
        data = np.c_[np.ones((data.shape[0], 1)), data]
        return np.dot(data, self.weights)
        