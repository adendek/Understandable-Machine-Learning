import numpy as np

class DummyOptim:
    def __init__(self):
        pass

    def optimize(self, weights, data, target):
        return np.arange(weights.shape[0])

