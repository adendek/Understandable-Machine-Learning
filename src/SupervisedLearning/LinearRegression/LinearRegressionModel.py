import numpy as np
from src.SupervisedLearning.LinearRegression.Optimizer import BatchGradientDecent
from src.SupervisedLearning.LinearRegression.Loss import MSE
from src.SupervisedLearning.LinearRegression.Validator import LinRegValidator


class LinearRegressionModel:
    def __init__(self, n_features, optimizer=None, loss=None):
        self.n_features = n_features
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = BatchGradientDecent(learning_rate=1, n_steps=100, save_history=True)
        if loss:
            self.loss = loss
        else:
            self.loss = MSE()        
        self.validator = LinRegValidator(n_features=n_features)
        
    def fit(self, data, target, weights=None):
        self.validator.validate_training(data, target)
        self.weights = self.optimizer.optimize(data, target, loss=self.loss, weights=weights)
        
    def predict(self, data):
        data = np.c_[np.ones((data.shape[0], 1)), data]
        return np.dot(data, self.weights)
        