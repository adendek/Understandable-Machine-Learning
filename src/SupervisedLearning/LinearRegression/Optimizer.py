import numpy as np
from Loss import MSE

class DummyOptim:
    def __init__(self):
        """
        This class is a default implementation of the Optimizer.
        It is used to test correctness of the LogisticRgressionModel implementation.
        """
        pass

    def optimize(self, data, target, loss, weights):
        return np.arange(weights.shape[0])
    
    
class BatchGradientDecent:
    def __init__(self, learning_rate = 1, n_steps = 10, save_history=False):
        """
        This is the default implementation of the Batch Gradient Decent algorithm.
        :param learning_rate: step size
        :param n_steps: number of optimization steps
        :param save_history: flag whether to save gradients and weights, that can be use to
        debug/analyze the learning progress
        """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.save_history = save_history
        if save_history:
            self.history = {} # dictionary that keeps track of the previously calculated gradients


    def optimize(self, data, target, loss=None, weights=None):
        if isinstance(weights, np.ndarray):
            pass
        else:
            weights = np.random.rand(data.shape[1] + 1, 1) # add weight for bias term 
        data = np.c_[np.ones((data.shape[0], 1)), data] # add bias term (x0 = 1) to each instance
        if not loss:
            loss = MSE()
        for step in range(self.n_steps):
            forward = loss._forward(weights, data)
            loss_value = loss._loss(forward, target)
            gradient = loss._grad(forward, weights, data, target)   
            weights = weights - self.learning_rate * gradient
            if self.save_history:
                self.__save_history(step, weights, loss_value, gradient)
        return weights


    def __save_history(self, step, weights, loss_value, gradient):
        self.history[step] = {"gradient": gradient,
                              "loss": loss_value,
                              "weights": weights
                              }

        
class StochasticGradientDecent:
    def __init__(self, learning_rate = 1, n_steps = 10, save_history=False):
        """
        This is the default implementation of the Stochastic Decent algorithm.
        :param learning_rate: step size
        :param n_steps: number of optimization steps
        :param save_history: flag whether to save gradients and weights, that can be use to
        debug/analyze the learning progress
        """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.save_history = save_history
        if save_history:
            self.history = {} # dictionary that keeps track of the previously calculated gradients


    def optimize(self, data, target, loss=None, weights=None):
        m = data.shape[0]
        if isinstance(weights, np.ndarray):
            pass
        else:
            weights = np.random.rand(data.shape[1] + 1, 1) # add weight for bias term 
        data = np.c_[np.ones((data.shape[0], 1)), data] # add bias term (x0 = 1) to each instance
#         if not weights:
#             weights = np.random.rand(data.shape[1] + 1, 1)
#         data = np.c_[np.ones((data.shape[0], 1)), data] # add bias term (x0 = 1) to each instance
        if not loss:
            loss = MSE()
        for step in range(self.n_steps):
            random_index = np.random.randint(0, m + 1)
            X = data[random_index:random_index+1]
            y = target[random_index:random_index+1]
            forward = loss._forward(weights, data)
            loss_value = loss._loss(forward, target)
            gradient = loss._grad(forward, weights, data, target)
            weights = weights - self.learning_rate * gradient
            if self.save_history:
                self.__save_history(step, weights, loss_value, gradient)
        return weights


    def __save_history(self, step, weights, loss_value, gradient):
        self.history[step] = {"gradient": gradient,
                              "loss": loss_value,
                              "weights": weights
                              }
