from .value_based_agent import ValueBasedAgent
from ..utils import ReplayBuffer
from ...utils.nn import MLP
import copy
import numpy as np
import torch


class MunchausenDQN(ValueBasedAgent):
    """
    An agent that learn an approximated Q-Function using a neural network.
    This Q-Function is used to find the best action to execute in a given observation.
    """

    def __init__(self, observation_space, action_space, **params):

        """
        Args:
            observation_space (Union[gym.spaces.Box, gym.spaces.Discrete]): The environment's observation space.
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete]): The environment's action space.
            name (str_, optional): The agent's name.
            device (torch.device_, optional): The device on which the agent operates.
            batch_size (int, optional): Number of samples in each training batch.
            replay_buffer_size (int, optional): Maximum number of interaction records the replay buffer can store
                before it is full.
            nb_gradient_steps (int, optional): Specifies how many times the learning process is repeated each time
                `agent.learn()` is called.
            gamma (float, optional): Discount factor used in the critic's target computation formula.
            layer_1_size (int, optional): Size of the first layer in the actor, critic, and target networks.
            layer_2_size (int, optional): Size of the second layer in the actor, critic, and target networks.
            epsilon_target (Union[float, int], optional): Epsilon term of the training target computation.
            model (torch.nn.module, optional): A model to replace the default Q-network. It is optional though.
            optimizer_class (Type[torch.optim.Optimizer], optional): The class of the optimizer to use, it will be
                instantiated using the model parameters and given learning rate.
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): Loss function.
            learning_rate (float, optional): Learning rate for the Q-network.
            tau (float, optional): Soft update coefficient for the target actor and critic networks.
            l_0 (float, optional): The lower bound of the weighted log probability
            tau_soft (float, optional): Attribute reducing the amplitude of the Q-network output, thus making its 
                entropy higher, and then increasing exploration. Default: 0.03.
            steps_before_target_update (int, optional): how many learning steps should we perform before to modify the 
                target networks weights.
            alpha (float, optional): The entropy regularization parameter.
        """

        super().__init__(observation_space, action_space, **params)
        self.name = params.get("name", "Munchausen DQN")
        self.batch_size = params.get("batch_size", 256)
        self.replay_buffer_size = params.get("replay_buffer_size", int(1e6))
        self.nb_gradient_steps = params.get("nb_gradient_steps", 1)
        self.gamma = params.get("gamma", 0.98)
        self.layer_1_size = params.get("layer_1_size", 128)
        self.layer_2_size = params.get("layer_2_size", 128)
        self.epsilon_target = params.get("epsilon_target", 1e-6)
        self.model = params.get("model", None)
        self.optimizer_class = params.get("optimizer_class", torch.optim.Adam)
        self.criterion = params.get("criterion", torch.nn.functional.smooth_l1_loss)
        self.learning_rate = params.get("learning_rate", 5e-4)
        self.tau = params.get("tau", 0.001)
        self.l_0 = params.get("l_0", -1.0)  # The lower bound of the weighted log probability
        self.tau_soft = params.get("tau_soft", 0.03)
        self.steps_before_target_update = params.get("steps_before_target_update", 1)
        self.alpha = params.get("alpha", 0.9)  # The entropy regularization parameter

        self.total_steps = 0
        self.steps_since_last_target_update = 0
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.device)
        if self.model is None:
            self.model = MLP(self.observation_size,
                             self.layer_1_size, torch.nn.ReLU(),
                             self.layer_2_size, torch.nn.ReLU(),
                             self.action_size, learning_rate=self.learning_rate,
                             optimizer_class=self.optimizer_class,
                             device=self.device).float()
        self.target_model = copy.deepcopy(self.model).to(self.device)

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
        assert observation.shape == self.observation_space.shape

        with torch.no_grad():
            q_values = self.model(observation)
            policy = torch.nn.functional.softmax(q_values / self.tau_soft, dim=-1)
            action = torch.multinomial(policy, 1).squeeze(-1).cpu().numpy()
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

        for _ in range(self.nb_gradient_steps):
            if len(self.replay_buffer) > self.batch_size:
                observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.batch_size)
                actions = actions.to(dtype=torch.int64)[:, np.newaxis]

                with torch.no_grad():
                    target_q_values = self.target_model(observations)
                    target_policy = torch.nn.functional.softmax(target_q_values / self.tau_soft, dim=-1)
                    target_next_q_values = self.target_model(next_observations)
                    target_next_policy = torch.nn.functional.softmax(target_next_q_values / self.tau_soft, dim=-1)
                    log_part = torch.log(target_policy.gather(1, actions).squeeze())
                    red_term = self.alpha * (self.tau_soft * log_part + self.epsilon_target).clamp(self.l_0, 0.0)
                    bleu_term = -self.tau_soft * torch.log(target_next_policy + self.epsilon_target)
                    munchausen_target = rewards + red_term + self.gamma * (1 - dones) * \
                        (target_next_policy * (target_next_q_values + bleu_term)).sum(dim=-1)
                    td_target = munchausen_target.squeeze()

                old_val = self.model(observations).gather(1, actions).squeeze()
                loss = torch.nn.functional.mse_loss(td_target, old_val)
                self.model.learn(loss)

        self.steps_since_last_target_update += 1
        if self.steps_since_last_target_update >= self.steps_before_target_update:

            self.target_model.converge_to(self.model, self.tau)
            self.steps_since_last_target_update = 0

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
