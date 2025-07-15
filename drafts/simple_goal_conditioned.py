from statistics import mean
from rlnav import PointMazeV0
from torch import optim

from sciborg import SAC, DDPG, TD3, HER, ConditioningWrapper

grid_size = 15
start_x = grid_size // 2
maze_array = [[1] * grid_size]
for _ in range(grid_size - 2):
    maze_array.append([1] + [0] * (grid_size - 2) + [1])
maze_array.append([1] * grid_size)
maze_array[start_x][start_x] = 2  # Start_tile

environment = PointMazeV0(goal_conditioned=True, maze_array=maze_array)

agent = HER(TD3, environment.observation_space, environment.action_space, goal_space=environment.goal_space,
            gamma=0.99,
            exploration_noise_std=0.2,
            policy_update_frequency=2,
            batch_size=256,
            replay_buffer_size=int(1e6),
            steps_before_learning=100,
            target_network_update_frequency=1,
            layer_1_size=256,
            layer_2_size=256,
            actor_tau=0.001,
            critic_tau=0.001,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=0.00025,
            alpha=0.2,
            actor_optimizer_class=optim.Adam,
            critic_optimizer_class=optim.Adam,
            autotune_alpha=True,
            target_entropy_scale=0.89)

results = []
for episode_id in range(100):
    print(f"Episode {episode_id} ... 0%", end="\r")
    observation, info = environment.reset()
    goal = info["goal"]
    agent.start_episode(observation, goal)
    result = 0
    for interaction_id in range(70):
        action = agent.action(observation)
        observation, reward, terminated, _, info = environment.step(action)
        agent.process_interaction(action, reward, observation, terminated, learn=True)
        print(f"Episode {episode_id} ... {round(interaction_id / 70 * 100)}%", end="\r")
        if terminated:
            result = 1
    agent.stop_episode()
    results.append(result)
    if len(results) > 20:
        print(f"Episode {episode_id} DONE. Result = {result}; Last 20 mean = {mean(results[-20:])}")
    else:
        print(f"Episode {episode_id} DONE. Result = {result}")
