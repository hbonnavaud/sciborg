from typing import Union
import numpy as np
from gymnasium.spaces import Box, Discrete

from ..conditioned.conditioned_agent import ConditionedAgent
from ... import RLAgent


class ConditioningWrapper(ConditionedAgent):
    """
    A wrapper that build a conditioned instance of the specified agent class.
    We call conditioned agent any RLAgent that have to choose action depending on the observation but also on a
    conditioning data. The said data could be a goal, a skill, or any information that could be added to the agent
    inputs.

    NB: You can ignore this class and condition the agent in your main function if you want.
    Those conditioned agent classes are made to implement some basic function such as concatenation between the
    conditioning data and the observation, at any step of the agent's life, in order to make yours easier.
    """
    def __init__(self,
                 reinforcement_learning_agent_class,
                 observation_space: Union[Box, Discrete],
                 action_space: Union[Box, Discrete],
                 conditioning_space: Union[Box, Discrete],
                 **params):

        """
        Args:
            reinforcement_learning_agent_class: The class of the agent to be wrapped.
            This class will be instantiated using the given parameters, and the computed information (such as the new
            conditioned observation space).
            observation_space: The wrapper observation space, WHICH IS NOT EQUAL TO THE WRAPPED AGENT OBSERVATION SPACE.
            action_space: The agent's action space.
            conditioning_space: The conditioning space.
            **params:
        """

        # Super class init + add to self.init_params, parameters that are not send to it.
        super().__init__(observation_space, action_space, conditioning_space, **params)
        self.name = params.get("name", "Conditioned " + self.reinforcement_learning_agent.name)

        # # Verify and store the wrapped agent class
        assert issubclass(reinforcement_learning_agent_class, RLAgent)
        self.reinforcement_learning_agent_class = reinforcement_learning_agent_class
        self.init_params["reinforcement_learning_agent_class"] = reinforcement_learning_agent_class

        # Compute feature space (observation space + conditioning space)
        if isinstance(self.observation_space, Box) and isinstance(self.conditioning_space, Box):
            self.feature_space = Box(
                low=np.concatenate((self.observation_space.low, self.conditioning_space.low), 0),
                high=np.concatenate((self.observation_space.high, self.conditioning_space.high), 0))
        elif isinstance(self.observation_space, Discrete) and isinstance(self.conditioning_space, Discrete):
            self.feature_space = Discrete(self.observation_space.n * self.conditioning_space.n)
        else:
            raise NotImplementedError("Observation space ang conditioning space with different types are not supported.")

        # Use this feature space as the wrapped agent's observation space in order to instantiate it.
        self.reinforcement_learning_agent: RLAgent = (
            reinforcement_learning_agent_class(self.feature_space, action_space, **params))


    def __getattr__(self, name):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.reinforcement_learning_agent, name)

    def start_episode(self, *episode_information, test_episode=False):
        """
        Args:
            episode_information: a tuple containing two instances:
             - information[0]: observation: The first observation of the episode.
             - information[1]: conditioning: The conditioning of the episode.
            test_episode: Boolean indication whether the episode is a test episode or not.
            If it is a test episode, the agent will not explore (fully deterministic actions) and not learn (no
            interaction data storage or learning process).
        """
        observation, conditioning = episode_information
        super().start_episode(observation, test_episode)
        self.current_conditioning = conditioning
        self.reinforcement_learning_agent.start_episode(self.get_features(observation, self.current_conditioning),
                                                        test_episode=test_episode)

    def get_features(self, observations, conditioning=None) -> np.ndarray:
        """
        Args:
            observations: The observations from which we want the feature (Could be a single one or a batch).
            conditioning: The conditioning from which we want the feature (Could be a single one or a batch).
        Returns: the features that will be used as an input for the agent's NNs.
        """

        # Both shapes should match exactly, but this function is supposed to work with a batch of observation and a
        # single goal if needed.
        assert len(observations.shape) < 3
        assert len(conditioning.shape) < 3
        observations_batch_size = 1 if observations.shape == self.observation_space.shape else observations.shape[0]
        conditioning_batch_size = 1 if conditioning.shape == self.conditioning_space.shape else conditioning.shape[0]

        # If observations are a batch and goal is a single one
        if observations_batch_size == 1 and conditioning_batch_size > 1:
            if observations.shape != self.observation_space.shape:
                observations = observations.squeeze()  # Remove batch
            observations = np.tile(observations, (conditioning_batch_size, *tuple(np.ones(len(observations.shape)).astype(int))))
        elif conditioning_batch_size == 1 and observations_batch_size > 1:
            if conditioning.shape != self.goal_space.shape:
                conditioning = conditioning.squeeze()  # Remove batch
            conditioning = np.tile(conditioning, (observations_batch_size, *tuple(np.ones(len(conditioning.shape)).astype(int))))

        if isinstance(self.observation_space, Box) and isinstance(self.goal_space, Box):
            return np.concatenate((observations, conditioning),
                                  axis=int(observations_batch_size > 1 or conditioning_batch_size > 1))
        elif isinstance(self.observation_space, Discrete) and isinstance(self.goal_space, Discrete):
            return observations + conditioning * self.observation_space.n
        else:
            raise NotImplementedError("Observation space ang goal space with different types are not supported.")

    def action(self, observation, conditioning=None, explore=True) -> np.ndarray:
        """
        Args:
            observation: the observation from which we want the agent to take an action.
            conditioning: the data conditioning the agent for this action. If this data is None, then the conditioning
            used is the one stored fo the episode, i.e. self.current_conditioning.
            explore: boolean indicating whether the agent can explore with this action of only exploit.
            If test_episode was set to True in the last self.start_episode call, the agent will exploit (explore=False)
            no matter the explore value here.
        Returns: the action chosen by the agent.
        """
        return self.reinforcement_learning_agent.action(
            self.get_features(observation, self.current_conditioning),
            explore)

    def learn(self):
        """
        Trigger the agent learning process.
        Make sure that self.test_episode is False, otherwise, an error will be raised.
        """
        self.reinforcement_learning_agent.learn()

    def process_interaction(self, action: np.ndarray, reward: float, new_observation: np.ndarray,
                            done: bool, learn: bool = True):
        """
        Processed the passed interaction using the given information.
        The state from which the action has been performed is kept in the agent's attribute, and updated everytime this function is called.
        Therefore, it does not appear in the function signature.
        Args:
            action (np.ndarray): the action performed by the agent at this step.
            reward (float): the reward returned by the environment following this action.
            new_observation (np.ndarray): the new state reached by the agent with this action.
            done (bool): whether the episode is done (no action will be performed from the given new_state) or not.
            learn (bool): whether the agent cal learn from this step or not (will define if the agent can save this interaction
                data, and start a learning step or not).
        """
        super().process_interaction(action, reward, new_observation, done, learn=learn)
        new_observation = self.get_features(new_observation, self.current_contitioning)
        self.reinforcement_learning_agent.process_interaction(action, reward, new_observation, done, learn)

    def set_device(self, device):
        super().set_device(device)
        self.reinforcement_learning_agent.set_device(device)
