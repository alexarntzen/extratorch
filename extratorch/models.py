from typing import Union
import torch
import torch.nn as nn

from extratorch.train import fit_module

activations = {
    "LeakyReLU": nn.LeakyReLU,
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}


class FFNN(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        neurons=None,
        activation: Union[str, callable] = "tanh",
        init=None,
        **kwargs,
    ):
        super(FFNN, self).__init__()

        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = activation
        if isinstance(activation, str):
            self.activation_func = activations[self.activation]
        elif callable(activation):
            self.activation_func = activation()
        else:
            raise ValueError(f"Activation {activation} not recognized")

        if neurons is not None:
            self.input_layer = nn.Linear(self.input_dimension, self.neurons)
            self.hidden_layers = nn.ModuleList(
                [
                    nn.Linear(self.neurons, self.neurons)
                    for _ in range(n_hidden_layers - 1)
                ]
            )
            self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        # init
        if init is not None:
            init(self)

    def forward(self, x):
        # The forward function performs the set of affine and
        # non-linear transformations defining the network
        # (see equation above)
        x = self.activation_func(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation_func(l(x))
        return self.output_layer(x)

    def __str__(self):
        """return name of class"""
        return type(self).__name__


def NeuralNet_Seq(input_dimension, output_dimension, n_hidden_layers, neurons):
    modules = list()
    modules.append(nn.Linear(input_dimension, neurons))
    modules.append(nn.Tanh())
    for _ in range(n_hidden_layers):
        modules.append(nn.Linear(neurons, neurons))
        modules.append(nn.Tanh())
    modules.append(nn.Linear(neurons, output_dimension))
    model = nn.Sequential(*modules)
    return model


def init_xavier(model, init_weight_seed=None, **kwargs):
    if init_weight_seed is not None:
        torch.manual_seed(init_weight_seed)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain(model.activation)
            # torch.nn.init.xavier_uniform_(m.weight, gain=g)
            nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)

    model.apply(init_weights)
    return model


def get_scaled_model(model, x_center=0, x_scale=1, y_center=0, y_scale=1):
    def scaled_model(x):
        return model((x - x_center) / x_scale) * y_scale + y_center

    return scaled_model


def get_trained_model(
    model_param,
    training_param,
    data,
    data_val=None,
    fit=fit_module,
):
    # Xavier weight initialization
    model = model_param.pop("model")(**model_param)
    model, history = fit(
        model=model,
        data=data,
        data_val=data_val,
        **training_param,
        model_param=model_param,
    )
    return model, history
