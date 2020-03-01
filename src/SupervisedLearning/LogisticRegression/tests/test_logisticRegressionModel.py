from unittest import TestCase
from src.SupervisedLearning.LogisticRegression.LogisticRegressionModel import *
from src.SupervisedLearning.LogisticRegression.Validator import *
from src.SupervisedLearning.LogisticRegression.Optimizer import DummyOptim
import numpy as np


class TestLogisticRegressionModel(TestCase):
    def setUp(self) -> None:
        self.n_features = 4
        self.lr = LogisticRegressionModel(self.n_features, n_class=2, optimizer=DummyOptim())

    def test_fit_with_dummy_optim(self):
        n_events = 5
        data = np.random.randint(10, size=(self.n_features, n_events))
        target = np.zeros(n_events)
        self.lr.fit(data, target)

    def test_fit_should_rise_if_wrong_data_dimensionality(self):
        n_events = 5
        wrong_dim = self.n_features-2
        data = np.random.randint(10, size=(wrong_dim, n_events))
        target = np.zeros(n_events)

        self.assertRaises(DataDimError,self.lr.fit,data,target)

    def test_fit_should_rise_if_data_target_not_equal_examples(self):
        n_events = 5
        data = np.random.randint(10, size=(self.n_features, n_events))
        target = np.zeros(n_events-1)

        self.assertRaises(DataTargetMissmatch,self.lr.fit,data,target)

    def test_predict_with_dummy_optim(self):
        n_events = 5
        data = np.random.randint(10, size=(self.n_features, n_events))
        predictions = self.lr.predict(data)
        self.assertEqual(predictions.shape[0], n_events)

