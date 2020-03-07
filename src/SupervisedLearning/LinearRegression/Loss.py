import numpy as np

class MSE():
    def __init__(self):
        pass
        
    def _grad(self, forward, weights, data, target):
        m = data.shape[0]
        return 2 / m * np.dot(data.T, forward - target)

    
    def _loss(self, forward, target):
        return np.square(np.subtract(forward, target)).mean()

    
    def _forward(self, weights, data):
        return np.dot(data, weights)