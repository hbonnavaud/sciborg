# Goal conditioned deep Q-network
import pickle
from .value_based_agent import ValueBasedAgent
from ..utils import ReplayBuffer
import copy
import numpy as np
import torch
from .value_based_agent import ValueBasedAgent
from ..utils import ReplayBuffer
from gymnasium.spaces import Discrete


class DQN(ValueBasedAgent):
    """
    An agent that learn an approximated Q-Function using a neural network.
    This Q-Function is used to find the best action to execute in a given observation.
    """

    OBSERVATION_SPACE_TYPE=Discrete

    def __init__(self, *args, **params):

        """
        Args:
            observation_space (Union[gym.spaces.Box, gym.spaces.Discrete]): The environment's observation space.
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete]): The environment's action space.
            name (str, optional): The agent's name.
            device (torch.device, optional): The device on which the agent operates.
            batch_size (int, optional): Number of samples in each training batch.
            replay_buffer_size (int, optional): Maximum number of interaction records the replay buffer can store
                before it is full.
            steps_before_learning (int, optional): Number of steps to take before learning begins, allowing the replay
                buffer to fill.
            learning_frequency (int, optional): Interval of learning steps. Learning is triggered every time the
                current step index modulo `learning_frequency` equals zero.
            nb_gradient_steps (int, optional): Specifies how many times the learning process is repeated each time
                `agent.learn()` is called.
            gamma (float, optional): Discount factor used in the critic's target computation formula.
            layer_1_size (int, optional): Size of the first layer in the actor, critic, and target networks.
            layer_2_size (int, optional): Size of the second layer in the actor, critic, and target networks.
            initial_epsilon (Union[float, int], optional): Initial value for the epsilon attribute.
            final_epsilon (Union[float, int], optional): Final (i.e. lowest) value for the epsilon attribute.
            steps_before_epsilon_decay (int, optional): How many action (in learning steps) have to be performed before
                epsilon starts to decay.
            epsilon_decay_period (int, optional): How long does the epsilon attribute have to decay before to reach its
                final value (in this implementation, the decay step is not given, but computed from all the other
                epsilon parameters).
            model (torch.nn.module, optional): A model to replace the default Q-network. It is optional though.
            optimizer_class (Type[torch.optim.Optimizer], optional): The class of the optimizer to use, it will be
                instantiated using the model parameters and given learning rate.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): Loss function.
            learning_rate (float, optional): Learning rate for the Q-network.
            tau (float, optional): Soft update coefficient for the target actor and critic networks.
        """

        super().__init__(*args, **params)
        self.name = params.get("name", "DQN")
        
        # Gather parameters
        self.batch_size = params.get("batch_size", 256)
        self.replay_buffer_size = params.get("replay_buffer_size", int(1e6))
        self.steps_before_learning = params.get("steps_before_learning", 0)
        self.learning_frequency = params.get("learning_frequency", 1)
        self.nb_gradient_steps = params.get("nb_gradient_steps", 1)
        self.gamma = params.get("gamma", 0.95)
        self.layer_1_size = params.get("layer_1_size", 128)
        self.layer_2_size = params.get("layer_2_size", 84)
        self.initial_epsilon = params.get("initial_epsilon", 1)
        self.final_epsilon = params.get("final_epsilon", 0.05)
        self.steps_before_epsilon_decay = params.get("steps_before_epsilon_decay", 20)
        self.epsilon_decay_period = params.get("epsilon_decay_period", 1000)
        self.model = params.get("model", None)
        self.optimizer_class = params.get("optimizer_class", torch.optim.Adam)
        self.criterion = params.get("criterion", torch.nn.functional.mse_loss)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.tau = params.get("tau", 0.0003)

        assert issubclass(self.optimizer_class, torch.optim.Optimizer)

        # Instantiate the class
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.device)
        self.epsilon = None
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.epsilon_decay_period
        self.total_steps = 0
        if self.model is None:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.observation_size, self.layer_1_size), torch.nn.ReLU(),
                torch.nn.Linear(self.layer_1_size, self.layer_2_size), torch.nn.ReLU(),
                torch.nn.Linear(self.layer_2_size, self.action_size)
            ).to(self.device)
        else:
            assert isinstance(self.model, torch.nn.Module)
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
        self.target_model = copy.deepcopy(self.model)

    def start_episode(self, observation, test_episode=False):
        """
        Args:
            observation (np.ndarray): The first observation of the episode.
            test_episode (bool, optional): Boolean indication whether the episode is a test episode or not.
            If it is a test episode, the agent will not explore (fully deterministic actions) and not learn (no
            interaction data storage or learning process).
        """
        self.epsilon = 1.0
        super().start_episode(observation, test_episode)

    def action(self, observation: np.ndarray, explore=True) -> np.ndarray:
        """
        Args:
            observation: The observation from which we want the agent to take an action.
            explore: Boolean indicating whether the agent can explore with this action of only exploit.
            If test_episode was set to True in the last self.start_episode call, the agent will exploit (explore=False)
            no matter the explore value here.
        Returns:
            np.ndarray: The action chosen by the agent.
        """
        if explore and not self.under_test and self.train_interactions_done > self.steps_before_epsilon_decay:
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_step)

        if explore and not self.under_test and np.random.rand() < self.epsilon:  # Epsilon greedy
            action = np.random.randint(self.action_space.n)
        else:
            # greedy_action(self.model, observation) function in RL5 notebook
            if isinstance(observation, np.ndarray):
                observation = torch.from_numpy(observation).to(self.device)
            with torch.no_grad():
                q_values = self.model(observation)
                action = torch.argmax(q_values).item()
        return action

    def process_interaction(self,
                            action: np.ndarray,
                            reward: float,
                            new_observation: np.ndarray,
                            done: bool,
                            learn: bool = True):
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
        if learn and not self.under_test:
            self.replay_buffer.append((self.last_observation, action, reward, new_observation, done))
            self.learn()
        super().process_interaction(action, reward, new_observation, done, learn=learn)

    def learn(self):
        """
        Trigger the agent learning process.
        Make sure that self.test_episode is False, otherwise, an error will be raised.
        """
        assert not self.under_test
        if (self.train_interactions_done >= self.steps_before_learning
                and self.train_interactions_done % self.learning_frequency == 0
                and len(self.replay_buffer) > self.batch_size):
            for _ in range(self.nb_gradient_steps):
                observations, actions, rewards, new_observations, dones = self.replay_buffer.sample(self.batch_size)
                q_prime = self.target_model(new_observations).max(1)[0].detach()
                q_target = rewards + self.gamma * (1 - dones) * q_prime
                q_values = self.model(observations).gather(1, actions.to(torch.long).unsqueeze(1))
                loss = self.criterion(q_values, q_target.unsqueeze(1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_value(self, observations: np.ndarray, actions: np.ndarray = None) -> np.ndarray:
        """
        Args:
            observations (np.ndarray): The observation(s) from which we want to obtain a value. Could be a batch.
            actions (np.ndarray, optional): The action that will be performed from the given observation(s). If none,
                the agent compute itself which action it would have taken from these observations.
        Returns:
            np.ndarray: The value of the given features.
        """
        with torch.no_grad():
            values = self.model(observations)
            if actions is None:
                values = values.max(-1).values
            else:
                if not isinstance(actions, torch.Tensor):
                    actions = torch.tensor(actions)
                values = values.gather(1, actions.to(torch.long).unsqueeze(1))
        return values.cpu().detach().numpy()
