import numpy as np
import torch
from torch import nn, optim
from torch.nn.modules import Module
from torch.optim import Optimizer


class NeuralNetwork(Module):
    """
    A general MLP class. Initialisation example:
    mlp = MLP(input_size, 64, ReLU(), 64, ReLU(), output_size, Sigmoid())
    """

    def __init__(self,
                 module: Module,
                 device=None,
                 tau: float = 0.1,
                 learning_rate: float = 0.01,
                 optimizer_class=optim.Adam):
        """
        For each element in layers_data:
         - If the element is an integer, it will be replaced by a linear layer with this integer as output size,
         - If this is a model (like activation layer) il will be directly integrated
         - If it is a function, it will be used to initialise the weights of the layer before
            So we call layer_data[n](layer_data[n - 1].weights) with n the index of the activation function in
            layers_data
        """
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.module: Module = module
        self.tau = tau
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = optimizer_class(params=self.parameters(), lr=learning_rate)
        assert isinstance(self.optimizer, Optimizer)
        self.to(self.device)

    def forward(self, input_data):
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data).to(self.device)
        input_data = input_data.float()
        return self.module(input_data)

    def converge_to(self, other_nn, tau=None):
        """
        Make the weights of the current model be a bit closer to the given mlp.
        self.weights = (1 - tau) * self.weights + tau * other_mlp.weights
        Precondition: other_mlp have the exact same shape of self.
        """
        tau = self.tau if tau is None else tau
        assert isinstance(tau, float)
        for self_param, other_param in zip(self.parameters(), other_nn.parameters()):
            self_param.data.copy_(
                self_param.data * (1.0 - tau) + other_param.data * tau
            )

    def learn(self, loss, retain_graph=False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()


class MLP(NeuralNetwork):
    """
    A general MLP class. Initialisation example:
    mlp = MLP(input_size, 64, ReLU(), 64, ReLU(), output_size, Sigmoid())
    """

    def __init__(self, input_size, *layers_data, device, tau=0.1, learning_rate=0.01,
                 optimizer_class=optim.Adam):
        """
        For each element in layers_data:
         - If the element is an integer, it will be replaced by a linear layer with this integer as output size,
         - If this is a model (like activation layer) il will be directly integrated
         - If it is a function, it will be used to initialise the weights of the layer before
            So we call layer_data[n](layer_data[n - 1].weights) with n the index of the activation function in
            layers_data
        """

        layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for data in layers_data:
            layer = data
            if isinstance(data, int):
                layer = nn.Linear(input_size, data)
                input_size = data
            if callable(data) and not isinstance(data, nn.Module):
                data(self.layers[-1].weight)
                continue
            layers.append(layer)  # For the next layer
        layers = nn.Sequential(*layers)

        super().__init__(layers, device=device, tau=tau, learning_rate=learning_rate, optimizer_class=optimizer_class)
