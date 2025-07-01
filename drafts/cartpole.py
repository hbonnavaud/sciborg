import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")  # Or "Qt5Agg", depending on what's installed
import gymnasium as gym
from sciborg import MunchausenDQN
from statistics import mean
from sciborg.utils import generate_video, create_dir

if __name__ == "__main__":

    # Initialisation
    render = False
    # nb_interactions = 70
    environment = gym.make("CartPole-v1", render_mode="rgb_array")
    # environment = PointMazeV0(goal_conditioned=True, action_noise=1.0)
    # agent = HER(TD3, environment.observation_space, environment.action_space, goal_space=environment.goal_space,
    #   layer_1_size=64, layer_2_size=64, learning_rate=3e-4, batch_size=128)
    print("Agent initialisation")
    agent = MunchausenDQN(environment.observation_space, environment.action_space)
    create_dir("generated_videos")
    results_mem = []
    results_running_avg = []
    for episode_id in range(5000):
        # if episode_id != 0 and episode_id % 100 == 0:
        #     plt.plot(agent.critic_1_losses, c='b')
        #     plt.plot(agent.critic_2_losses, c='g')
        #     plt.plot(agent.actor_losses, c='r')
        #     plt.show()
        #     print(agent.critic_1_losses[-100:])
        #     print(agent.critic_2_losses[-100:])
        observation, info = environment.reset()
        if render:
            images = [environment.render()]
        agent.start_episode(observation, test_episode=False)
        result = 0
        reward_sum = 0
        truncated = terminated = False
        while not (truncated or terminated):
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
        agent.stop_episode()

        results_mem.append(reward_sum)
        if len(results_mem) >= 20:
            results_running_avg.append(mean(results_mem[-20:]))
            print(f"Episode {episode_id}; rewards sum: {reward_sum}; last 20 avg.: {results_running_avg[-1]}")
        else:
            print(f"Episode {episode_id}; rewards sum: {reward_sum}")
        if render:
            generate_video(images, "generated_videos", f"episode_{episode_id}.mp4")
    plt.plot(results_running_avg)
    plt.title("running_average")
    plt.show()