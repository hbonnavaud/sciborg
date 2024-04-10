import copy
import json
import os.path
import pickle
from typing import Union

import gym
from gym.spaces import Box, Discrete
import torch
from abc import ABC, abstractmethod

from utils import create_dir


class Agent(ABC):
    """
    A global agent class that describe the interactions between our agent, and it's environment.
    """

    name = "Agent"

    def __init__(self, observation_space: Union[Box, Discrete], action_space: Union[Box, Discrete], **params):
        """
        @param observation_space: Environment's observation space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """
        assert isinstance(observation_space, Box) or isinstance(observation_space, Discrete), \
            "The observation space should be an instance of gym.spaces.Space"
        assert isinstance(action_space, Box) or isinstance(action_space, Discrete), \
            "The action space should be an instance of gym.spaces.Space"

        self.init_params = params
        self.init_params["observation_space"] = observation_space   # used for copy
        self.init_params["action_space"] = action_space
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = params.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Mandatory parameters
        assert not isinstance(self.observation_space, dict), "observation space as dictionary is not supported."
        self.observation_size = self.observation_space.shape[0]  # Assume observation space is continuous
        self.observation_shape = self.observation_space.shape

        self.continuous = isinstance(self.action_space, gym.spaces.Box)
        self.nb_actions = self.action_space.shape[0] if self.continuous else self.action_space.n
        self.last_observation = None  # Useful to store interaction when we receive (new_stare, reward, done) tuple
        self.episode_id = 0
        self.episode_time_step_id = 0
        self.training_steps_done = 0
        self.output_dir = None
        self.under_test = False
        self.episode_started = False

    def start_episode(self, *information, test_episode=False):
        """
        *information instead of observation alone because child classes can put a lot of stuff at this place,
        like skill id, goal to reach, or whatever people imagination think about.
        """
        (observation,) = information
        if self.episode_started:
            self.stop_episode()
        self.episode_started = True
        self.last_observation = observation
        self.episode_time_step_id = 0
        self.under_test = test_episode

    def process_interaction(self, action, reward, new_observation, done, learn=True):
        self.episode_time_step_id += 1
        if learn and not self.under_test:
            self.training_steps_done += 1
        self.last_observation = new_observation

    def stop_episode(self):
        self.episode_id += 1
        self.episode_started = False

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

    def reset(self):
        self.__init__(**self.init_params)

    def copy(self):
        return copy.deepcopy(self)

    def save(self, directory):
        create_dir(directory)
        save_in_json = {}
        save_in_pickle = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                torch.save(v.observation_dict(), os.path.join(directory, k + ".pt"))
            if isinstance(v, (int, str)):
                # Store in a json file every variable that might be easy to read by a human, so we can easily access to
                # some information without building a python script
                save_in_json[k] = v
            else:
                # We consider the others attributes as too complex to be read by human (do you really want to take a
                # look in a replay buffer without a script?) so we will store them in a binary file to save space
                # and because some of them cannot be saved in a json file like numpy arrays.
                save_in_pickle[k] = v
        json.dump(save_in_json, open(os.path.join(directory, "basic_attributes.json"), "w"))
        pickle.dump(save_in_pickle, open(os.path.join(directory, "objects_attributes.pk"), "wb"))

    def load(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError("Directory " + directory + " not found or is not a directory.")

        # Load attributes in basic_variables.json
        try:
            saved_in_json = json.load(open(os.path.join(directory, "basic_attributes.json"), "r"))
            for k, v in saved_in_json.items():
                self.__setattr__(k, v)
        except FileNotFoundError:
            pass

        # Load attributes in objects_attributes.pk
        try:
            saved_in_pickle = pickle.load(open(os.path.join(directory, "objects_attributes.pk"), "rb"))
            for k, v in saved_in_pickle.items():
                self.__setattr__(k, v)
        except FileNotFoundError:
            pass

        # Load the rest
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                self.__getattribute__(k).load_observation_dict(torch.load(os.path.join(directory, k + ".pt")))

    @abstractmethod
    def action(self, observation, explore=True):
        pass

    @abstractmethod
    def set_device(self, device):
        pass
