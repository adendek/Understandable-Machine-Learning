import numpy as np

class BatchGradientDecent:
    def __init__(self, learning_rate = 1, n_steps = 10, save_history=False):
#    def __init__(self, learning_rate = 1, n_steps = 10, save_history=False):
        """
        This is the default implementation of the Gradient Decent algorithm.

        :param learning_rate: step size
        :param weights: Initial theta/weights parameters
        :param n_steps: number of optimization steps
        :param save_history: flag whether to save gradients and weights, that can be use to
        debug/analyze the learning progress
        """
        self.learning_rate = learning_rate

        self.n_steps = n_steps
        self.save_history = save_history

        if save_history:
            self.history = {} # dictionary that keeps track of the previously calculated gradients

#     def optimize(self, weights, data, target):
    def optimize(self, data, target, measure="MSE", weights=None):
        if not weights:
            weights = np.random.rand(data.shape[1] + 1, 1)
        data = np.c_[np.ones((data.shape[0], 1)), data] # add bias term (x0 = 1) to each instance
        for step in range(self.n_steps):
            forward = self.__forward(weights, data)
            loss = self.__loss(forward, target)
            gradient = self.__grad(weights, forward, data, target, measure)
            weights = weights - self.learning_rate * gradient
            if self.save_history:
                self.__save_history(step, weights, loss, gradient)

        return weights

    def __grad(self, weights, forward, data, target, measure):
        """
        TBD
        """
        if measure == "MSE":
            m = data.shape[0]
#             print("data.shape ", data.shape)
#             print("data.T.shape ", data.T.shape)
#             print("weights.shape ", weights.shape)
#             print("weights.T.shape ", weights.T.shape)
#             print("target.shape ", target.shape)
#             print("target.T.shape ", target.T.shape)
#             print(weights.shape)
#             print(target.shape)
            #return 2 / m * np.dot(data.T, np.dot(data.T, weights) - target)
            return 2 / m * data.T.dot(data.dot(weights) - target)

    
    def __loss(self, forward, target):
        return np.square(np.subtract(forward, target)).mean()
    
    def __forward(self, weights, data):
        return np.dot(data, weights)

    def __save_history(self, step, weights, loss, gradient):
        self.history[step] = {"gradient": gradient,
                              "loss":loss,
                              "new_weights": weights
                              }
