import gymnasium
from sciborg import SAC, TD3, DIAYN, MunchausenDQN
from rlnav import PointMazeV0, GridWorldV0
from typing import Optional
import numpy as np
from sciborg.utils import save_image
from statistics import mean


skills_colors = [
    [255, 0, 0],      # Red
    [255, 64, 0],     # Orange-red
    [255, 128, 0],    # Orange
    [255, 192, 0],    # Yellow-orange
    [255, 255, 0],    # Yellow
    [192, 255, 0],    # Yellow-green
    [128, 255, 0],    # Lime
    [0, 255, 0],      # Green
    [0, 255, 128],    # Spring green
    [0, 255, 255],    # Cyan
    [0, 128, 255],    # Sky blue
    [0, 0, 255],      # Blue
    [128, 0, 255],    # Violet
    [255, 0, 255],    # Magenta / Purple
]


def episode(
    agent: DIAYN, 
    environment: PointMazeV0, 
    forced_skill: Optional[int] = None, 
    nb_interactions: int = 100, 
    learn: bool = True,
    render_image: Optional[np.ndarray] = None):
  

  assert forced_skill is None or isinstance(forced_skill, int)
  
  observation, info = environment.reset()
  agent.start_episode(observation, forced_skill=forced_skill, test_episode=not learn)
  for interaction_id in range(nb_interactions):
    action = agent.action(observation)
    new_observation, reward, terminated, truncated, info = environment.step(action)
    agent.process_interaction(action, reward, new_observation, terminated)
    if render_image is not None:
        if isinstance(environment, GridWorldV0):
            environment.set_tile_color(render_image, skills_colors[forced_skill], new_observation)
        else:
            environment.place_point(render_image, skills_colors[forced_skill], *new_observation[:2], radius=1)
  return render_image

def render_skills(agent: DIAYN, environment: PointMazeV0):
  render_image = environment.render(show_agent=False, show_rewards=False)
  for skill_id in range(agent.nb_skills):
    for episode_id in range(10):
      render_image = episode(agent, environment, forced_skill=skill_id, nb_interactions=100, learn=False, render_image=render_image)
  return render_image

if __name__ == "__main__":


    env = "grid_world"

    # Initialisation
    if env == "grid_world":

        room_size = 50
        maze_array = np.zeros((room_size, room_size))
        maze_array[room_size // 2, room_size // 2] = 2
        maze_array[0, :] = 1
        maze_array[-1, :] = 1
        maze_array[:, 0] = 1
        maze_array[:, -1] = 1

        environment = GridWorldV0(maze_array=maze_array, goal_conditioned=False, reset_anywhere=False)
        wrapped_agent_class = MunchausenDQN
    else:
        environment = PointMazeV0(goal_conditioned=False, action_noise=0.0)
        wrapped_agent_class = TD3

    agent = DIAYN(observation_space=environment.observation_space,
                action_space=environment.action_space,
                wrapped_agent_class=wrapped_agent_class,
                nb_skills=10)

    # Training
    steps_per_episodes = 100
    mean_intrinsic_rewards = []
    mean_discriminator_losses = []
    for episode_id in range(5000):
        intrinsic_reward_sum = 0
        nb_rewards = 0
        discriminator_loss_sum = 0
        nb_discriminator_loss = 0
        print(f"Episode {episode_id} ...", end="\r")
        observation, info = environment.reset()
        agent.start_episode(observation)
        for interaction_id in range(steps_per_episodes):
            print(f"Episode {episode_id} ... step {interaction_id} ({interaction_id / steps_per_episodes * 100}%).", end="\r")
            action = agent.action(observation)
            new_observation, reward, terminated, truncated, info = environment.step(action)
            intrinsic_reward, discriminator_loss = agent.process_interaction(action, reward, new_observation, terminated)
            if intrinsic_reward:
                intrinsic_reward_sum += intrinsic_reward
                nb_rewards += 1
            if discriminator_loss:
                discriminator_loss_sum += discriminator_loss
                nb_discriminator_loss += 1
        if nb_discriminator_loss > 0:
            mean_reward = intrinsic_reward_sum / nb_rewards
            mean_intrinsic_rewards.append(float(mean_reward))

            mean_losses = discriminator_loss_sum / nb_discriminator_loss
            mean_discriminator_losses.append(float(mean_losses))

            if len(mean_intrinsic_rewards) >= 20:
                print(f"Episode {episode_id} DONE. Average intrinsic reward = {mean_reward}, last 20: {mean(mean_intrinsic_rewards[-20:])}; "
                      f"average discriminator loss = {mean_losses}, last 20: {mean(mean_discriminator_losses)}.")
            else:
                print(f"Episode {episode_id} DONE. Average intrinsic reward = {mean_reward}; "
                      f"average discriminator loss = {mean_losses}.")
        else:
            print(f"Episode {episode_id} DONE.")

        if episode_id % 500 == 0:
            # Verification
            image = render_skills(agent, environment)
            save_image(image, "outputs", "Learned_skills_ep_" + str(episode_id))

    # Verification
    image = render_skills(agent, environment)
    save_image(image, "outputs", "final_skills")
