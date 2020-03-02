from src.SupervisedLearning.LogisticRegression.utils import *

class DummyOptim:
    def __init__(self):
        """
        This class is a default implementation of the Optimizer.
        It is used to test correctness of the LogisticRgressionModel implementation.
        """
        pass

    def optimize(self, weights, data, target):
        return np.arange(weights.shape[0])


class GradientDecent:
    def __init__(self, learning_rate = 1,  batch_size = -1, n_steps = 10, save_history=False):
        """
        This is the default implementation of the Gradient Decent algorithm.

        :param learning_rate: step size
        :param batch_size: Number of events to process during one update. If -1 use all of the examples
        :param n_steps: number of optimization steps
        :param save_history: flag whether to save gradients and weights, that can be use to
        debug/analyze the learning progress
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.save_history = save_history

        if save_history:
            self.history = {} # dictionary that keeps track of the previously calculated gradients

    def optimize(self, weights, data, target):
        for step in range(self.n_steps):
            forward = self.__forward(weights, data)
            loss = self.__loss(forward, target )
            gradient = self.__grad(weights, forward, data, target)
            weights = weights - self.learning_rate * gradient
            if self.save_history:
                self.__save_history(step, weights,loss, gradient)

        return weights

    def __grad(self, weights, forward, data, target):
        """
        This function return gradient of CrossEntropy with respect to the weights.
        Current implementation use predefined gradient, every time, when user wants to change the loss function,
        for instance by adding regularization term, he/she need to implement his/her own version of this function.
        """
        return np.dot(data.transpose(), (forward - target))/target.shape[0]
    
    def __loss(self, forward, target):
        return np.dot(forward, target) + np.dot(forward, (1-target))
   
    def __forward(self, weights, data):
        return sigmoid(np.dot(data, weights))

    def __save_history(self, step, weights, loss, gradient):
        self.history[step] = {"gradient": gradient,
                              "loss":loss,
                              "new_weights": weights
                              }
