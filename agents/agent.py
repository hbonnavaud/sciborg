import copy
import json
import os.path
import pickle
from typing import Union
import inspect
import gym
import numpy as np
from gym.spaces import Box, Discrete
import torch
from abc import ABC, abstractmethod
from ..utils import create_dir


class Agent(ABC):
    """
    A global agent class that describe the interactions between our agent, and it's environment.
    """

    name = "Agent"

    def __init__(self,
                 observation_space: Union[Box, Discrete],
                 action_space: Union[Box, Discrete],
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        """
        @param observation_space: Environment's observation space.
        @param action_space: Environment's action_space.
        @param device: agent's device.
        """
        assert isinstance(observation_space, Box) or isinstance(observation_space, Discrete), \
            "The observation space should be an instance of gym.spaces.Space"
        assert isinstance(action_space, Box) or isinstance(action_space, Discrete), \
            "The action space should be an instance of gym.spaces.Space"

        # Get the init parameters
        self.init_params = {}
        frame = inspect.stack()[0][0]
        while frame is not None and "self" in frame.f_locals and frame.f_locals["self"] == self:
            self.init_params.update(frame.f_locals)
            frame = frame.f_back
        for ignored in ["self", "frame", "__class__"]:
            self.init_params.pop(ignored)

        self.device = device

        # Observation info
        self.observation_space = observation_space
        assert isinstance(observation_space, (Box, Discrete)), "observation space type is not supported."
        if isinstance(observation_space, Box):
            self.observation_size = np.prod(self.observation_space.shape)
        else:
            assert isinstance(observation_space, Discrete)
            self.observation_size = observation_space.n

        # Action info
        self.action_space = action_space
        assert isinstance(action_space, (Box, Discrete)), "action space type is not supported."
        if isinstance(self.action_space, gym.spaces.Box):
            self.action_size = np.prod(self.action_space.shape)

        self.last_observation = None  # Useful to store interaction when we receive (new_stare, reward, done) tuple
        self.episode_id = 0
        self.episode_time_step_id = 0
        self.train_interactions_done = 0
        self.output_dir = None
        self.under_test = False
        self.episode_started = False

    def start_episode(self, *episode_info, test_episode=False):
        (observation,) = episode_info
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
                self.__getattribute__(attr).to(self.device)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, Agent):
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

    def save(self, directory):
        create_dir(directory)
        save_in_json = {}
        save_in_pickle = {"class": self.__class__}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                torch.save(v.state_dict(), str(directory / (k + ".pt")))
            elif isinstance(v, (int, str)):
                # Store in a json file every variable that might be easy to read by a human, so we can easily access to
                # some information without building a python script
                save_in_json[k] = v
            elif callable(v) and v.__name__ == "<lambda>":
                pass  # We cannot save lambda expression
            elif isinstance(v, Agent):
                v.save(directory / k)  # Wrapped agent recursive save
            else:
                # We consider the others attributes as too complex to be read by human (do you really want to take a
                # look in a replay buffer without a script?) so we will store them in a binary file to save space
                # and because some of them cannot be saved in a json file like numpy arrays.
                save_in_pickle[k] = v
        json.dump(save_in_json, open(str(directory / "basic_attributes.json"), "w"))
        pickle.dump(save_in_pickle, open(str(directory / "objects_attributes.pk"), "wb"))

    def load(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError("Directory " + directory + " not found or is not a directory.")

        # Load attributes in basic_variables.json
        try:
            saved_in_json = json.load(open(str(directory / "basic_attributes.json"), "r"))
            for k, v in saved_in_json.items():
                self.__setattr__(k, v)
        except FileNotFoundError:
            pass

        # Load attributes in objects_attributes.pk
        try:
            saved_in_pickle = pickle.load(open(str(directory / "objects_attributes.pk"), "rb"))
            for k, v in saved_in_pickle.items():
                if k == "class":
                    continue
                self.__setattr__(k, v)
        except FileNotFoundError:
            pass

        # Load models and wrapped agents
        for k, v in self.__dict__.items():
            if isinstance(v, Agent):
                v.load(directory / k)
            if isinstance(v, torch.nn.Module):
                self.__getattribute__(k).load_state_dict(torch.load(str(directory / (k + ".pt"))))

    @abstractmethod
    def action(self, observation, explore=True):
        pass
