import numpy as np
import torch
from torch import nn, optim
from typing import Type, Union


class NeuralNetwork(nn.Module):
    """
    A general MLP class. Initialisation example:
    mlp = MLP(input_size, 64, ReLU(), 64, ReLU(), output_size, Sigmoid())
    """

    def __init__(self,
                 tau: float = 0.1,
                 learning_rate: float = 0.01,
                 optimizer_class: Type[optim.Optimizer] = optim.Adam,
                 module: Union[None, nn.Module] = None):
        """
        This class add some stuff to the torch.nn.Module class in order to make the manipulation of rl NNs more concise.
        Args:
            tau: Used when the weights of this nn converge to the weights of another given one (used when this nn is a
                target network): self.weights = (1 - tau) * self.weights + tau * other_nn.weights
            learning_rate: Learning rate of the optimiser.
            optimizer_class: Class of the optimiser to use, an instance is created on self.__init__().
        """
        super().__init__()
        self.module = self if module is None else module
        self.tau = tau
        self.learning_rate = learning_rate
        assert issubclass(optimizer_class, optim.Optimizer)
        self.optimizer = optimizer_class(params=self.module.parameters(), lr=self.learning_rate)

    @property
    def device(self):
        return next(self.module.parameters()).data.device

    @property
    def dtype(self):
        return next(self.module.parameters()).data.dtype

    def format_input(self, input_data: Union[torch.Tensor, np.ndarray]):
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data).to(self.device)
        return input_data.to(self.dtype)

    def match(self, other_nn: nn.Module, tau: Union[int, float] = None):
        """
        Make the weights of the current model match the ones of the given mlp, with a ratio of tau.
        self.weights = (1 - tau) * self.weights + tau * other_mlp.weights
        Precondition: other_mlp's parameters have the exact same shape than self's ones.

        Args:
            other_nn: The other neural network to copy
            tau: default=None, the copy ratio. If not set, self.tau is used.
        """

        tau = self.tau if tau is None else tau
        for self_param, other_param in zip(self.module.parameters(), other_nn.parameters()):
            self_param.data.copy_(
                self_param.data * (1.0 - tau) + other_param.data * tau
            )

    def learn(self, loss: torch.Tensor, retain_graph: bool = False):
        """
        Backward the given loss on this module's weights.
        Args:
            loss: The loss to back-propagate.
            retain_graph: Whether the gradient graph should be retained or not.
        """
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
