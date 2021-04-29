import unittest
import torch

import torch.utils
import torch.utils.data

import numpy as np

from deepthermal.FFNN_model import get_trained_nn_model, FFNN, fit_FFNN, init_xavier
from deepthermal.validation import create_subdictionary_iterator, get_rMSE, k_fold_CV_grid


class TestOnSimpleFunctionApprox(unittest.TestCase):
    @staticmethod
    def exact_solution(x):
        return torch.sin(x)

    @classmethod
    def setUpClass(cls):
        n_samples = 1000
        sigma = 0.0

        cls.x = 2 * np.pi * torch.rand((n_samples, 1))
        cls.y = cls.exact_solution(cls.x) * (1 + sigma * torch.randn(cls.x.shape))
        cls.x_test = torch.linspace(0, 2 * np.pi, 10000).reshape(-1, 1)
        cls.y_test = cls.exact_solution(cls.x_test)

    def test_approx_error(self):
        model_params = {
            "input_dimension": 1,
            "output_dimension": 1,
            "n_hidden_layers": 3,
            "neurons": 10,
            "activation": "relu",
        }
        training_params = {
            "num_epochs": 1000,
            "batch_size": 500,
            "regularization_exp": 2,
            "regularization_param": 1e-4,
            "optimizer": "ADAM",
            "init_weight_seed": 10
        }

        model = get_trained_nn_model(model_params, training_params, self.x, self.y)

        rel_test_error = get_rMSE(model, self.x_test, self.y_test)
        self.assertAlmostEqual(0, rel_test_error, delta=0.01)

    def test_k_fold_CV_grid(self):
        model_params = {
            "input_dimension": [1],
            "output_dimension": [1],
            "n_hidden_layers": [3],
            "neurons": [10, 20],
            "activation": ["relu", ]
        }
        training_params = {
            "num_epochs": [1000],
            "batch_size": [500],
            "regularization_exp": [2],
            "regularization_param": [1e-4],
            "optimizer": ["ADAM"],
            "init_weight_seed": [10]

        }

        model_params_iterator = create_subdictionary_iterator(model_params)
        training_params_iterator = create_subdictionary_iterator(training_params)

        models, rel_train_errors, rel_val_errors = k_fold_CV_grid(FFNN, model_params_iterator, fit_FFNN,
                                                                  training_params_iterator, self.x, self.y,
                                                                  init=init_xavier,
                                                                  folds=3)
        avg_rel_val_errors = torch.mean(torch.tensor(rel_val_errors), dim=1)
        self.assertAlmostEqual(0, torch.max(avg_rel_val_errors).item(), delta=0.01)

        for submodels in models:
            for model in submodels:
                rel_test_error = get_rMSE(model, self.x_test, self.y_test)
                self.assertAlmostEqual(0, rel_test_error, delta=0.01)

    def test_k_fold_CV_grid_partial(self):
        model_params = {
            "input_dimension": [1],
            "output_dimension": [1],
            "n_hidden_layers": [3],
            "neurons": [10, 20],
            "activation": ["relu"],
        }
        training_params = {
            "num_epochs": [1000],
            "batch_size": [500],
            "regularization_exp": [2],
            "regularization_param": [1e-4],
            "optimizer": ["ADAM"],
            "init_weight_seed": [20]
        }

        model_params_iterator = create_subdictionary_iterator(model_params)
        training_params_iterator = create_subdictionary_iterator(training_params)

        models, rel_train_errors, rel_val_errors = k_fold_CV_grid(FFNN, model_params_iterator, fit_FFNN,
                                                                  training_params_iterator, self.x, self.y,
                                                                  init=init_xavier,
                                                                  folds=5, partial=True)

        avg_rel_val_errors = torch.mean(torch.tensor(rel_val_errors), dim=1)
        self.assertAlmostEqual(0, torch.max(avg_rel_val_errors).item(), delta=0.01)

        for submodels in models:
            for model in submodels:
                rel_test_error = get_rMSE(model, self.x_test, self.y_test)
                self.assertAlmostEqual(0, rel_test_error, delta=0.01)


if __name__ == '__main__':
    unittest.main()
