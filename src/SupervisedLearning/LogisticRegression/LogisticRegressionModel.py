import numpy as np
from src.SupervisedLearning.LogisticRegression.Validator import LogRegValidator
from src.SupervisedLearning.LogisticRegression.utils import *


class LogisticRegressionModel:
    def __init__(self, n_features, n_class=2, optimizer="GD"):
        self.weights = np.zeros(n_features)
        self.optimizer = optimizer
        self.n_class = n_class
        self.validator = LogRegValidator(n_features=n_features, n_classes=n_class)

    def fit(self, data, target):
        self.validator.validate_training(data, target)
        self.weights = self.optimizer.optimize(self.weights, data, target)

    def predict(self, to_predict):
        return sigmoid(np.dot(self.weights.T,to_predict))
