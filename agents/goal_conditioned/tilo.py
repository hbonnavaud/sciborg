import numpy as np
from gym.spaces import Box
from agents.goal_conditioned.her import HER


class TILO(HER):
    """
    A global agent class for goal conditioned agents. The # NEW tag indicate differences between Agent class and this
    one.
    """

    def __init__(self,
                 reinforcement_learning_agent_class,
                 observation_space,
                 action_space,
                 goal_from_observation_fun,
                 **params):

        assert isinstance(observation_space, Box), "The observation space should be an instance of gym.spaces.Box. " \
                                             "Discrete observation space is not supported."
        self.goal_space = params.get("goal_space", observation_space)
        if params.get("get_features", None) is not None:
            self.get_features = params.get("get_features", None)
            assert hasattr(self.get_features, "__call__")  # Make sure it is a function
        else:
            # Otherwise we need a observation to goal projection for the default "get_features" function.
            self.observation_to_goal_filter = params.get("observation_to_goal_filter", None)
            if self.observation_to_goal_filter is None:
                self.observation_to_goal_filter = np.zeros(observation_space.shape).astype(int)
                self.observation_to_goal_filter[np.where(np.ones(self.goal_space.shape))] = 1

        super().__init__(reinforcement_learning_agent_class, observation_space, action_space,
                         goal_from_observation_fun, **params)

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
        if len(observations.shape) == 1:
            a = self.get_goal_from_observation(observations)
            b = goals
            observation_goal_diff = goals - self.get_goal_from_observation(observations)
            features[:self.goal_shape[0]] = observation_goal_diff
        else:
            observation_goal_diff = goals - observations[:, self.observation_to_goal_filter]
            features[:, self.goal_shape[0]] = observation_goal_diff
        return features
