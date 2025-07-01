import copy
import json
import os.path
import pickle
from typing import Union
import inspect
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
import torch
from abc import ABC, abstractmethod
from ..utils import create_dir


class RLAgent(ABC):
    """
    A global agent class that describe the interactions between our agent, and it's environment.
    """

    name = "Agent"

    def __init__(self,
                 observation_space: Union[Box, Discrete],
                 action_space: Union[Box, Discrete],
                 **params):
        """
        @param observation_space: Environment's observation space.
        @param action_space: Environment's action_space.
        @param device: agent's device.
        """
        assert isinstance(observation_space, (Box, Discrete)), \
            "The observation space should be an instance of gym.spaces.Space"
        assert isinstance(action_space, (Box, Discrete)), \
            "The action space should be an instance of gym.spaces.Space"
        
        self.device = params.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        assert isinstance(self.device, torch.device)
        
        # assert isinstance
        # Get the init parameters
        self.init_params = {}
        frame = inspect.stack()[1][0]  # Fixed: should be stack()[1] not stack()[0]
        # OLD: frame = inspect.stack()[0][0]
        while frame is not None and "self" in frame.f_locals and frame.f_locals["self"] == self:
            self.init_params.update(frame.f_locals)
            frame = frame.f_back
        for ignored in ["self", "frame", "__class__"]:
            self.init_params.pop(ignored, None)  # Fixed: use pop with default
            # OLD: self.init_params.pop(ignored)
        print("computed init params: ", self.init_params)

        # Observation info
        self.observation_space = observation_space
        assert isinstance(observation_space, (Box, Discrete)), "observation space type is not supported."
        if isinstance(observation_space, Box):
            self.observation_size = np.prod(self.observation_space.shape)
        else:
            self.observation_size = observation_space.n

        # Action info
        self.action_space = action_space
        if isinstance(self.action_space, Box):  # Fixed: was gym.spaces.Box
            self.action_size = np.prod(self.action_space.shape)
        else:
            self.action_size = int(action_space.n)  # Added for discrete actions

        self.last_observation = None
        self.episode_id = 0
        self.episode_time_step_id = 0
        self.train_interactions_done = 0
        self.output_dir = None
        self.under_test = False
        self.episode_started = False

    def start_episode(self, observation, test_episode=False):  # Simplified signature
        if self.episode_started:
            self.stop_episode()
        self.episode_started = True
        self.last_observation = observation
        self.episode_time_step_id = 0
        self.under_test = test_episode

    def process_interaction(self, action, reward, new_observation, done, learn=True):
        self.episode_time_step_id += 1
        if learn and not self.under_test:
            self.train_interactions_done += 1
        self.last_observation = new_observation

    def stop_episode(self):
        self.episode_id += 1
        self.episode_started = False

    def set_device(self, device):
        self.device = device
        for attr, value in vars(self).items():
            if isinstance(value, torch.nn.Module):
                getattr(self, attr).to(self.device)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, RLAgent):  # Fixed: was Agent
                setattr(result, k, copy.deepcopy(v))
            elif isinstance(v, dict):
                new_dict = {}
                for k_, v_ in v.items():
                    new_dict[k_] = v_.copy() if k_ == "goal_reaching_agent" else copy.deepcopy(v_)
                setattr(result, k, new_dict)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def copy(self):
        return copy.deepcopy(self)

    def save(self, directory: str):
        create_dir(directory)
        save_in_json = {}
        save_in_pickle = {"class": self.__class__}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                torch.save(v.state_dict(), os.path.join(directory, k + ".pt"))
            elif isinstance(v, (int, str, float, bool)):
                # Store in a json file every variable that might be easy to read by a human, so we can easily access to
                # some information without building a python script
                save_in_json[k] = v
            elif callable(v) and hasattr(v, '__name__') and v.__name__ == "<lambda>":
                pass  # Cannot save lambda expressions
            elif isinstance(v, RLAgent):
                v.save(os.path.join(directory, k))
            else:
                # We consider the others attributes as too complex to be read by human (do you really want to take a
                # look in a replay buffer without a script?) so we will store them in a binary file to save space
                # and because some of them cannot be saved in a json file like numpy arrays.
                save_in_pickle[k] = v
        
        with open(os.path.join(directory, "basic_attributes.json"), "w") as f:
            json.dump(save_in_json, f)
        with open(os.path.join(directory, "objects_attributes.pk"), "wb") as f:
            pickle.dump(save_in_pickle, f)

    def load(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory {directory} not found or is not a directory.")

        # Load basic attributes
        try:
            with open(os.path.join(directory, "basic_attributes.json"), "r") as f:
                saved_in_json = json.load(f)
                for k, v in saved_in_json.items():
                    setattr(self, k, v)
        except FileNotFoundError:
            pass

        # Load complex attributes
        try:
            with open(os.path.join(directory, "objects_attributes.pk"), "rb") as f:
                saved_in_pickle = pickle.load(f)
                for k, v in saved_in_pickle.items():
                    if k == "class":
                        continue
                    setattr(self, k, v)
        except FileNotFoundError:
            pass

        # Load models and wrapped agents
        for k, v in self.__dict__.items():
            if isinstance(v, RLAgent):  # Fixed: was Agent
                v.load(os.path.join(directory, k))
            elif isinstance(v, torch.nn.Module):
                model_path = os.path.join(directory, k + ".pt")
                if os.path.exists(model_path):
                    getattr(self, k).load_state_dict(torch.load(model_path, map_location=self.device))

    @abstractmethod
    def action(self, observation, explore=True):
        pass