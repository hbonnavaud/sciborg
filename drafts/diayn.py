from sciborg import SAC, DIAYN
from rlnav import PointMazeV0
from typing import Optional
import numpy as np
from sciborg.utils import save_image


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
  
  observation = environment.reset()
  agent.start_episode(observation, forced_skill=forced_skill, test_episode=not learn)
  for interaction_id in range(nb_interactions):
    action = agent.action(observation)
    new_observation, reward, done, info = environment.step(action)
    agent.process_interaction(action, reward, new_observation, done)
    if render_image is not None:
      environment.place_point(render_image, skills_colors[forced_skill], new_observation[:2], radius=1)
  return render_image

def render_skills(agent: DIAYN, environment: PointMazeV0):
  render_image = environment.render(show_agent=False, show_rewards=False)
  for skill_id in range(agent.nb_skills):
    for episode_id in range(10):
      render_image = episode(agent, environment, forced_skill=skill_id, nb_interactions=100, learn=False, render_image=render_image)
  return render_image

if __name__ == "__main__":

  # Initialisation
  environment = PointMazeV0(goal_conditioned=False, action_noise=0.0)
  agent = DIAYN(observation_space=environment.observation_space, 
                action_space=environment.action_space, 
                agent_class=SAC)
  
  # Training
  for episode_id in range(1000):
    observation = environment.reset()
    agent.start_episode(observation)
    for interaction_id in range(100):
      action = agent.action(observation)
      new_observation, reward, done, info = environment.step(action)
      agent.process_interaction(action, reward, new_observation, done)
  
  # Verification
  image = render_skills(agent, environment)
  save_image(image, "outputs", "final_skills")
