import unittest
import torch
from torch.utils.data import TensorDataset
import numpy as np

from deepthermal.multilevel import MultilevelDataset, get_level_dataset, predict_multilevel, fit_multilevel_FFNN, \
    get_trained_multilevel_model
from deepthermal.validation import get_RRSE


class TestPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_samples = 2000
        cls.x = 2 * np.pi * torch.rand((cls.n_samples, 1))
        cls.y_0 = torch.sin(cls.x)
        cls.y_1 = cls.y_0[:cls.n_samples // 2]
        cls.y_2 = cls.y_0[:cls.n_samples // 4]
        cls.datalist = [cls.y_0.detach().clone(), cls.y_1.detach().clone(), cls.y_2.detach().clone()]
        print(cls.datalist[1].shape)
        cls.len_list = [cls.n_samples, cls.n_samples // 2, cls.n_samples // 4]

    def test_dataset_params(self):
        print("\n\n Test whether MultilevelDataset stores the right data and return the right dataset:")
        dataset = MultilevelDataset(self.datalist, self.x)

        x_data, y_data = dataset[:]
        for l in range(len(y_data)):
            self.assertEqual(x_data.size(0), y_data[l].size(0))
            self.assertEqual(torch.sum(torch.isnan(y_data[l])), self.n_samples - self.len_list[l])
            x_train, y_train = get_level_dataset(x_data, y_data, l)
            self.assertEqual(x_train.size(0), y_train.size(0))
            self.assertEqual(torch.max(torch.abs(self.x[:self.len_list[l]] - x_train)).item(), 0)
            if l == 0:
                self.assertEqual(torch.max(torch.abs(y_train - self.y_0)).item(), 0)
            elif l > 0:
                self.assertEqual(
                    torch.max(torch.abs(
                        y_train - self.datalist[l] + self.datalist[l - 1][:self.len_list[l]]
                    )).item()
                    , 0)

    def test_multilevel_approx_error(self):
        print("\n\n Approximating the sine function with a multilevel model:")
        ml_dataset = MultilevelDataset(self.datalist, self.x)
        model_params = {
            "input_dimension": 1,
            "output_dimension": 1,
            "n_hidden_layers": 5,
            "neurons": 10,
            "activation": "relu",
        }
        training_params = {
            "num_epochs": 500,
            "batch_size": 500,
            "regularization_exp": 2,
            "regularization_param": 1e-6,
            "optimizer": "ADAM",
            "learning_rate": 0.01,
            "init_weight_seed": 20
        }

        model, loss_history_train, loss_history_val = get_trained_multilevel_model(model_params,
                                                                                   training_params,
                                                                                   multilevel_data=ml_dataset,
                                                                                   fit=fit_multilevel_FFNN
                                                                                   )

        rel_test_error = get_RRSE(model, TensorDataset(self.x, self.y_0), predict=predict_multilevel)
        self.assertAlmostEqual(0, rel_test_error, delta=0.1)


if __name__ == '__main__':
    unittest.main()
