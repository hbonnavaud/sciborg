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
from collections import deque
import random
from ..rl_agent import RLAgent
from ..value_based_agents import SAC, MunchausenDQN, ValueBasedAgent
from sciborg.utils import one_hot

class Discriminator(torch.nn.Module):
    """Discriminator network that predicts skill from state"""
    def __init__(self, state_dim, nb_skill, hidden_dim=256):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, nb_skill)
        
    def forward(self, observation):
        x = torch.relu(self.fc1(observation))
        x = torch.relu(self.fc2(x))
        return torch.log_softmax(self.fc3(x), dim=-1)


class ReplayBuffer:
    """Simple replay buffer for storing discriminator training data"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, skill_idx):
        self.buffer.append((state, skill_idx))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        states, skill_indices = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(skill_indices)
        )
    
    def __len__(self):
        return len(self.buffer)


class DIAYN(RLAgent):
    """
    DIAYN (Diversity is All You Need) wrapper for RL agents.
    Learns diverse skills by maximizing mutual information between skills and states.
    """
    
    name = "DIAYN"
    
    def __init__(self, *args, **params):
        """
        @param nb_skills: Number of discrete skills to learn
        @param discriminator_lr: Learning rate for the discriminator
        @param discriminator_update_freq: How often to update discriminator (in interactions)
        @param buffer_size: Size of replay buffer for discriminator training
        @param batch_size: Batch size for discriminator updates
        @param wrapped_agent_params: Keyword arguments to pass to the agent constructor
        @param wrapped_agent_class: Class of the RL agent to wrap (must inherit from RLAgent)
        @param device: Device to run on
        """
        super().__init__(*args, **params)

        self.nb_skills = params.get("nb_skills", 10)
        self.discriminator_lr = params.get("discriminator_lr", 0.003)
        self.discriminator_update_freq = params.get("discriminator_update_freq", 1)
        self.buffer_size = params.get("buffer_size", 10000)
        self.batch_size = params.get("batch_size", 125)
        self.wrapped_agent_params= params.get("wrapped_agent_params", {})
        self.wrapped_agent_class = params.get("wrapped_agent_class", SAC if isinstance(self.observation_space, Box) else MunchausenDQN)
        assert issubclass(self.wrapped_agent_class, ValueBasedAgent)

        # Compute the wrapped agent observation space
        if isinstance(self.observation_space, Box):
            aug_obs_space = Box(
                low=np.concatenate([self.observation_space.low.flatten(), np.zeros(self.nb_skills)]),
                high=np.concatenate([self.observation_space.high.flatten(), np.ones(self.nb_skills)]),
                dtype=self.observation_space.dtype
            )
        elif isinstance(self.observation_space, Discrete):
            aug_obs_space = Discrete(self.observation_space.n + self.nb_skills)
        else:
            raise NotImplementedError("DIAYN currently only supports Box and Discrete observation spaces")

        # Force set some wrapped agent's params values
        self.wrapped_agent_params["device"] = self.device  # Have to be on the same device
        self.wrapped_agent = self.wrapped_agent_class(aug_obs_space, self.action_space, **self.wrapped_agent_params)

        self.discriminator = Discriminator(self.observation_size, self.nb_skills).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator_lr
        )
        
        # Replay buffer for discriminator training
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # Current skill and episode state
        self.current_skill = None
        self.current_skill_onehot = None
        self.discriminator_update_counter = 0
        
    def _sample_skill(self, forced_skill=None):
        """Sample a random skill"""
        assert forced_skill is None or isinstance(forced_skill, int)
        skill_idx = forced_skill if forced_skill is not None else np.random.randint(0, self.nb_skills)
        skill_onehot = np.zeros(self.nb_skills)
        skill_onehot[skill_idx] = 1.0
        return skill_idx, skill_onehot
    
    def _augment_observation(self, observation):
        """Augment observation with current skill"""
        if isinstance(self.observation_space, Box):
            obs_flat = observation.flatten()
            return np.concatenate([obs_flat, self.current_skill_onehot])
        else:
            return np.concatenate([one_hot(observation, self.observation_space.n), self.current_skill_onehot])
    
    def _compute_diayn_reward(self, next_observation):
        """Compute DIAYN pseudo-reward: log q(z|s') - log p(z)"""
        if isinstance(self.observation_space, Box):
            observation_tensor = torch.FloatTensor(next_observation.flatten()).unsqueeze(0).to(self.device)
        elif isinstance(self.observation_space, Discrete):
            observation_tensor = torch.FloatTensor([one_hot(next_observation, self.observation_size)]).to(self.device)
        
        with torch.no_grad():
            log_probs = self.discriminator(observation_tensor)
            # Reward is log q(z|s') - log p(z), where p(z) is uniform
            reward = log_probs[0, self.current_skill].item() - np.log(1.0 / self.nb_skills)
        
        return reward
    
    def _update_discriminator(self):
        """Update discriminator to predict skill from state"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, skill_indices = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        skill_indices = skill_indices.to(self.device)
        
        # Forward pass
        log_probs = self.discriminator(states)
        
        # Compute loss (negative log likelihood)
        loss = torch.nn.functional.nll_loss(log_probs, skill_indices)
        
        # Backward pass
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        return loss
    
    def start_episode(self, observation, test_episode=False, forced_skill=None):
        """Start new episode with a new skill"""
        assert forced_skill is None or isinstance(forced_skill, int)
        # Sample new skill for this episode
        self.current_skill, self.current_skill_onehot = self._sample_skill(forced_skill=forced_skill)
        
        # Augment observation and start episode for wrapped agent
        augmented_obs = self._augment_observation(observation)
        self.wrapped_agent.start_episode(augmented_obs, test_episode)
        
        # Call parent start_episode
        super().start_episode(observation, test_episode)
    
    def action(self, observation, explore=True):
        """Get action from wrapped agent using augmented observation"""
        augmented_obs = self._augment_observation(observation)
        return self.wrapped_agent.action(augmented_obs, explore)
    
    def process_interaction(self, action, env_reward, new_observation, done, learn=True):
        """Process interaction with DIAYN reward computation"""
        # Compute DIAYN pseudo-reward
        if learn and not self.under_test:
            diayn_reward = self._compute_diayn_reward(new_observation)
            
            # Store state-skill pair for discriminator training
            if isinstance(self.observation_space, Box):
                state_for_buffer = new_observation.flatten()
            else:
                state_for_buffer = one_hot(new_observation, self.observation_size)
            self.replay_buffer.push(state_for_buffer, self.current_skill)
            
            # Update discriminator periodically
            self.discriminator_update_counter += 1
            if self.discriminator_update_counter % self.discriminator_update_freq == 0:
                discriminator_loss = self._update_discriminator()
        else:
            diayn_reward = None
            discriminator_loss = None
        
        # Pass DIAYN reward to wrapped agent instead of environment reward
        augmented_new_obs = self._augment_observation(new_observation)
        if diayn_reward:
            self.wrapped_agent.process_interaction(action, diayn_reward, augmented_new_obs, done, learn)
        else:
            self.wrapped_agent.process_interaction(action, 0.0, augmented_new_obs, done, learn=False)
        
        # Call parent process_interaction with original env reward for logging
        super().process_interaction(action, env_reward, new_observation, done, learn)

        return diayn_reward, discriminator_loss
    
    def stop_episode(self):
        """Stop episode for both wrapper and wrapped agent"""
        self.wrapped_agent.stop_episode()
        super().stop_episode()
    
    def set_device(self, device):
        """Set device for both wrapper and wrapped agent"""
        super().set_device(device)
        self.wrapped_agent.set_device(device)
        self.discriminator.to(device)
    
    def save(self, directory):
        """Save both wrapper and wrapped agent"""
        super().save(directory)
        self.wrapped_agent.save(os.path.join(directory, "wrapped_agent"))
        # Discriminator is saved automatically as it's a torch.nn.Module
    
    def load(self, directory):
        """Load both wrapper and wrapped agent"""
        super().load(directory)
        self.wrapped_agent.load(os.path.join(directory, "wrapped_agent"))
    
    def get_skill_policy(self, skill_idx):
        """Get a policy that always uses a specific skill"""
        class SkillPolicy:
            def __init__(self, diayn_agent, skill_idx):
                self.diayn_agent = diayn_agent
                self.skill_idx = skill_idx
                self.skill_onehot = np.zeros(diayn_agent.nb_skill)
                self.skill_onehot[skill_idx] = 1.0
            
            def action(self, observation, explore=True):
                # Temporarily set the skill
                old_skill = self.diayn_agent.current_skill
                old_skill_onehot = self.diayn_agent.current_skill_onehot
                
                self.diayn_agent.current_skill = self.skill_idx
                self.diayn_agent.current_skill_onehot = self.skill_onehot
                
                action = self.diayn_agent.action(observation, explore)
                
                # Restore old skill
                self.diayn_agent.current_skill = old_skill
                self.diayn_agent.current_skill_onehot = old_skill_onehot
                
                return action
        
        return SkillPolicy(self, skill_idx)


# Example usage
class SimpleAgent(RLAgent):
    """Example agent implementation for testing"""
    name = "SimpleAgent"
    
    def __init__(self, observation_space, action_space, lr=1e-3, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        
        # Simple policy network
        if isinstance(observation_space, Box):
            input_dim = np.prod(observation_space.shape)
        else:
            input_dim = observation_space.n
            
        if isinstance(action_space, Box):
            output_dim = np.prod(action_space.shape)
            self.continuous = True
        else:
            output_dim = action_space.n
            self.continuous = False
        
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
    def action(self, observation, explore=True):
        obs_tensor = torch.FloatTensor(observation).to(self.device)
        with torch.no_grad():
            if self.continuous:
                action = self.policy(obs_tensor).cpu().numpy()
                if explore:
                    action += np.random.normal(0, 0.1, action.shape)
                return np.clip(action, self.action_space.low, self.action_space.high)
            else:
                logits = self.policy(obs_tensor)
                if explore:
                    action = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
                else:
                    action = torch.argmax(logits).item()
                return action


def demo_diayn():
    """Demonstration of DIAYN usage"""
    # Create environment
    env = gym.make('CartPole-v1')  # Simple environment for demo
    
    # Agent configuration
    agent_kwargs = {
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'lr': 1e-3
    }
    
    # Create DIAYN wrapper
    diayn_agent = DIAYN(
        agent_class=SimpleAgent,
        agent_kwargs=agent_kwargs,
        nb_skill=4,  # Learn 4 different skills
        discriminator_lr=3e-4,
        discriminator_update_freq=10
    )
    
    print(f"Created DIAYN agent with {diayn_agent.nb_skill} skills")
    print(f"Augmented observation space: {diayn_agent.observation_space}")
    
    # Train for a few episodes
    for episode in range(10):
        obs = env.reset()
        diayn_agent.start_episode(obs)
        
        total_reward = 0
        for step in range(200):
            action = diayn_agent.action(obs, explore=True)
            new_obs, reward, done, _ = env.step(action)
            
            diayn_agent.process_interaction(action, reward, new_obs, done)
            total_reward += reward
            obs = new_obs
            
            if done:
                break
        
        diayn_agent.stop_episode()
        print(f"Episode {episode}: Skill {diayn_agent.current_skill}, Reward: {total_reward}")
    
    # Demonstrate skill-specific policies
    print("\nTesting skill-specific policies:")
    for skill_idx in range(diayn_agent.nb_skill):
        skill_policy = diayn_agent.get_skill_policy(skill_idx)
        obs = env.reset()
        actions = []
        for _ in range(10):
            action = skill_policy.action(obs)
            actions.append(action)
            obs, _, done, _ = env.step(action)
            if done:
                break
        print(f"Skill {skill_idx} actions: {actions}")
    
    return diayn_agent

if __name__ == "__main__":
    # Run demonstration
    agent = demo_diayn()
    print("DIAYN wrapper implementation complete!")