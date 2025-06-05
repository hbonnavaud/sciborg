import pickle
import random
import numpy as np
import torch
import os
import json


def to_tensor(data_list, device):
    if isinstance(data_list[0], np.ndarray):
        data_list = np.array(data_list)
    return torch.Tensor(data_list).to(device=device)


class ReplayBuffer:
    def __init__(self, capacity, device, same_size_data=True):
        assert device is not None
        self.capacity = int(capacity)  # capacity of the buffer
        self.device = device
        self.data = []
        self.index = 0  # index of the next cell to be filled
        self.same_size_data = same_size_data
        if self.same_size_data:
            self.data_size = None

    def append(self, data: tuple):
        if self.same_size_data and self.data_size is None:
            self.data_size = len(data)
        elif self.same_size_data:
            assert self.data_size == len(data), "Impossible to store data of size " + str(len(data)) + " inside " \
                                                "buffer with data of size " + str(self.data_size) + "."
        assert isinstance(data[0], np.ndarray)

        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = data
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.data))
        batch = random.sample(self.data, batch_size)
        try:
            return list(map(lambda x: to_tensor(x, self.device), list(zip(*batch))))
        except Exception as e:
            result = []
            for elt_id, elt in enumerate(zip(*batch)):
                try:
                    result.append(to_tensor(elt, self.device))
                except Exception as exc:
                    print()
            print()

    def __len__(self):
        return len(self.data)
