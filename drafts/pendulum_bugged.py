import matplotlib

matplotlib.use("TkAgg")  # Or "Qt5Agg", depending on what's installed
import matplotlib.pyplot as plt
import gymnasium as gym
from sciborg import ValueBasedAgent, HER, SAC, TD3, DDPG, GoalConditionedWrapper
from rlnav import PointMazeV0
from statistics import mean
from sciborg.utils import generate_video, create_dir

if __name__ == "__main__":

    # Initialisation
    render = False
    # nb_interactions = 70
    environment = gym.make("Pendulum-v1", render_mode="rgb_array")
    # environment = PointMazeV0(conditioned=True, action_noise=1.0)
    # agent = HER(TD3, environment.observation_space, environment.action_space, goal_space=environment.goal_space,
    #   layer_1_size=64, layer_2_size=64, learning_rate=3e-4, batch_size=128)
    agent = TD3(environment.observation_space, environment.action_space)
    create_dir("generated_videos")
    results_mem = []
    for episode_id in range(500):
        if episode_id != 0 and episode_id % 100 == 0:
            plt.plot(agent.critic_1_losses, c='b')
            plt.plot(agent.critic_2_losses, c='g')
            plt.plot(agent.actor_losses, c='r')
            plt.show()
            print(agent.critic_1_losses[-100:])
            print(agent.critic_2_losses[-100:])
        observation, info = environment.reset()
        if render:
            images = [environment.render()]
        agent.start_episode(observation, test_episode=False)
        result = 0
        reward_sum = 0
        truncated = False
        while not truncated:
            # for interaction_id in range(nb_interactions):
            action = agent.action(observation)
            observation, reward, terminated, truncated, info = environment.step(action)
            reward_sum += reward
            # if interaction_id == nb_interactions - 1:
            #     terminated = True
            agent.process_interaction(action, reward, observation, terminated)
            # assert isinstance(agent, HER)
            if render:
                images.append(environment.render())
            if terminated or truncated:
                break
        agent.stop_episode()

        results_mem.append(reward_sum)
        if len(results_mem) >= 20:
            print(f"Episode {episode_id}; rewards sum: {reward_sum}; last 20 avg.: {mean(results_mem[-20:])}")
        else:
            print(f"Episode {episode_id}; rewards sum: {reward_sum}")
        if render:
            generate_video(images, "generated_videos", f"episode_{episode_id}.mp4")