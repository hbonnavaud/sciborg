import numpy as np
from typing import Union
from gym.spaces import Box, Discrete
from agents.goal_conditioned.her import HER


class TILO(HER):
    """
    A global agent class for goal conditioned agents. The # NEW tag indicate differences between Agent class and this
    one.
    """

    def __init__(self,
                 reinforcement_learning_agent_class,
                 observation_space: Union[Box, Discrete],
                 action_space: Union[Box, Discrete],
                 goal_space: Union[Box, Discrete] = None,
                 goal_from_observation_fun=None,
                 get_features=None,
                 **params):
        """
        :param reinforcement_learning_agent_class: The class of an agent that will be wrapped by this instance.
        :param observation_space: gym.spaces.Space observation space.
        :param action_space: gym.spaces.Space, action space.
        :param goal_from_observation_fun: A function that takes an observation and returns the goal that belongs to it.
            aka. a projection of the state space into the goal space.
        :param get_features: A function that build, from an observation and a goal, features that will be given to the
            wrapped agent. Those features should make the wrapped agent's control policy TILO.
        :param goal_space: gym.spaces.Space goal space.
        """

        # Initialise self.get_features
        if get_features is not None:
            assert hasattr(get_features, "__call__"), "The get_features argument should be a function."
            self.get_features = get_features

        # Super class init + add to self.init_params, parameters that are not send to it.
        super().__init__(reinforcement_learning_agent_class, observation_space, action_space, goal_space=goal_space,
                         goal_from_observation_fun=goal_from_observation_fun, **params)
        self.init_params["get_features"] = get_features

        # Modify instance name
        self.name = self.reinforcement_learning_agent.name + " + TILO"

    @property
    def feature_space(self):
        if isinstance(self.observation_space, Box):
            shape = self.get_features(self.observation_space.sample(), self.goal_space.sample()).shape
            return Box(low=np.full(shape, float("-inf")), high=np.full(shape, float("inf")))
        else:
            return self.observation_space

    def get_features(self, observations, goals):
        features = observations.copy()
        observation_as_goal = self.goal_from_observation_fun(observations)
        observation_goal_diff = goals - observation_as_goal
        features[..., :self.goal_shape[0]] = observation_goal_diff
        return features
