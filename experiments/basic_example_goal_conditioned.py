from statistics import mean
import matplotlib.pyplot as plt
from agents import DQN, HER, TILO
from environments import GoalConditionedGridWorld, GridWorldMapsIndex


environment = GoalConditionedGridWorld(map_name=GridWorldMapsIndex.EMPTY.value)
agent = HER(DQN, environment.observation_space, environment.action_space, environment.get_goal_from_observation,
            batch_size=250, learning_rate=0.0005, gamma=0.99)

results = []
results_running_average = []
for episode_id in range(200):

    obs, goal = environment.reset()
    agent.start_episode(obs, goal)
    result = 0
    for interaction_id in range(90):
        action = agent.action(obs)
        obs, reward, reached, _ = environment.step(action)
        agent.process_interaction(action, reward, obs, reached)
        if reached:
            result = 1
            break
    agent.stop_episode()

    results.append(result)
    if len(results) > 20:
        results_running_average.append(mean(results[-20:]))
        print("Episode ", episode_id, "; result=", result, " last 20 average = ", results_running_average[-1], sep="")
    else:
        print("Episode ", episode_id, "; result=", result, sep="")

plt.plot(results_running_average)
plt.show()
