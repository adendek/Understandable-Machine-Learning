import sys, os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from unittest import TestCase
from LinearRegressionModel import *
from Validator import *
from Optimizer import DummyOptim
from Loss import MSE
import numpy as np


class TestLinearRegressionModel(TestCase):
    def setUp(self):
        self.n_features = 4
        self.linreg = LinearRegressionModel(self.n_features, optimizer=DummyOptim(), loss=MSE())

    def test_fit_with_dummy_optim(self):
        n_events = 5
        data = np.random.randint(10, size=(n_events, self.n_features))
        target = np.zeros(n_events)
        self.linreg.fit(data, target)

    def test_fit_should_rise_if_wrong_data_dimensionality(self):
        n_events = 5
        wrong_dim = self.n_features-2
        data = np.random.randint(10, size=(n_events, wrong_dim))
        target = np.zeros(n_events)

        self.assertRaises(DataDimError, self.linreg.fit, data, target)

    def test_fit_should_rise_if_data_target_not_equal_examples(self):
        n_events = 5
        data = np.random.randint(10, size=(n_events, self.n_features))
        target = np.zeros(n_events-1)

        self.assertRaises(DataTargetMissmatch, self.linreg.fit, data, target)

    def test_predict_with_dummy_optim(self):
        n_events = 5
        data = np.random.randint(10, size=(n_events, self.n_features))
        predictions = self.linreg.predict(data)
        
        self.assertEqual(predictions.shape[0], n_events)
