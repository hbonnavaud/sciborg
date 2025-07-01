import matplotlib
matplotlib.use("TkAgg")  # Or "Qt5Agg", depending on what's installed
import matplotlib.pyplot as plt

from sciborg import ValueBasedAgent, HER, SAC, TD3, DDPG, GoalConditionedWrapper
from rlnav import PointMazeV0
from statistics import mean
from sciborg.utils import generate_video, create_dir, save_image
import numpy as np
import torch
from sciborg.utils import color_from_ratio


def render_value_function(agent: SAC, environment: PointMazeV0):
  assert environment.goal_conditioned

  # Value function computation
  step = 0.2
  X, Y = np.mgrid[environment.observation_space.low[0]:environment.observation_space.high[0] + step:step,
        environment.observation_space.low[1]:environment.observation_space.high[1] + step:step]
  xy = np.vstack((X.flatten(), Y.flatten())).T
  values = agent.get_value(xy, environment.goal)
  values = (values - values.min()) / (values.max() - values.min())

  # Drawing
  image = environment.render(show_rewards=False)
  for index, point in enumerate(xy):
    environment.place_point(image, color_from_ratio(values[index], hexadecimal=False), *point, radius=2)
  environment.place_point(image, [0, 0, 255], *environment.goal, radius=5)
  return image


if __name__ == "__main__":

  # Initialisation
  render = True
  render_val_fun = True
  nb_interactions = 70
  environment = PointMazeV0(goal_conditioned=True, action_noise=1.0)
  # agent = HER(TD3, environment.observation_space, environment.action_space, goal_space=environment.goal_space,
  #   layer_1_size=64, layer_2_size=64, learning_rate=3e-4, batch_size=128)
  agent = HER(SAC, environment.observation_space, environment.action_space, goal_space=environment.goal_space,
    layer_1_size=64, layer_2_size=64, learning_rate=3e-4, batch_size=128)
  create_dir("generated_videos")
  results_mem = []
  for episode_id in range(500):
    # if episode_id % 50 == 0:
    #   plt.plot(agent.reinforcement_learning_agent.critic_losses, c='b')
    #   plt.plot(agent.reinforcement_learning_agent.actor_losses, c='r')
    #   plt.show()

    observation, info = environment.reset()
    if render:
      images = [environment.render()]
    goal = info["goal"]
    agent.start_episode(observation, goal, test_episode=False)
    result = 0
    for interaction_id in range(nb_interactions):
      action = agent.action(observation)
      observation, reward, terminated, truncated, info = environment.step(action)
      if interaction_id == nb_interactions - 1:
        terminated = True
      agent.process_interaction(action, reward, observation, terminated)
      # assert isinstance(agent, HER)
      if render:
        images.append(environment.render())
      if info["reached"]:
        result = 1
        break
    agent.stop_episode()

    if render_val_fun and isinstance(agent.reinforcement_learning_agent, SAC):
      image = render_value_function(agent, environment)
      save_image(image, "value_function_visualisation", f"episode_{episode_id}")
    
    results_mem.append(1 if result else 0)
    if len(results_mem) >= 20:
      print(f"Episode {episode_id}; reached: {result}; last 20 avg.: {mean(results_mem[-20:])}")
    else:
      print(f"Episode {episode_id}; reached: {result}")
    if render:
      generate_video(images, "generated_videos", f"episode_{episode_id}.mp4")