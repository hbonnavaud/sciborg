from typing import Tuple, List

from rlnav import PointMazeV0
from sciborg import ConditionedAgent, ConditioningWrapper, SAC, HER, DDPG, TD3
from statistics import mean

grid_size = 17
start_x = grid_size // 2
maze_array = [[1] * grid_size]
for _ in range(grid_size - 2):
    maze_array.append([1] + [0] * (grid_size - 2) + [1])
maze_array.append([1] * grid_size)
maze_array[start_x][start_x] = 2  # Start_tile

environment = PointMazeV0(maze_array=maze_array, goal_conditioned=True, action_noise=0.0, sparse_reward=True)
agents = [
    ConditioningWrapper(SAC, environment.observation_space, environment.action_space, environment.goal_space),
    ConditioningWrapper(DDPG, environment.observation_space, environment.action_space, environment.goal_space),
    ConditioningWrapper(TD3, environment.observation_space, environment.action_space, environment.goal_space),
    HER(SAC, environment.observation_space, environment.action_space, environment.goal_space),
    HER(DDPG, environment.observation_space, environment.action_space, environment.goal_space),
    HER(TD3, environment.observation_space, environment.action_space, environment.goal_space)
]

def episode(
        agent: ConditionedAgent,
        env,
        test_episode: bool = False,
        nb_steps: int = 70,) -> Tuple[int, bool]:
    observation, info = env.reset()
    goal = info["goal"]
    agent.start_episode(observation, goal, test_episode=test_episode)

    result = False
    terminated = False
    steps_made = 0
    while not terminated:
        action = agent.action(observation)
        observation, reward, done, _, info = env.step(action)
        agent.process_interaction(action, reward, observation, done)
        steps_made += 1
        if terminated:
            result = True
        elif steps_made >= nb_steps:
            break

    agent.stop_episode()
    return result if test_episode else steps_made

def evaluation(
        agent: ConditionedAgent,
        env,
        tests_duration: int = 70,
        nb_tests: int = 10) -> float:
    nb_successes = 0
    for test_id in range(nb_tests):
        if episode(agent, env, test_episode=True, nb_steps=tests_duration):
            nb_successes += 1
    return nb_successes / nb_tests

def simulation(
        agent: ConditionedAgent,
        env,
        steps_per_episode: int = 70,
        steps_between_two_eval: int = 500,
        simulation_duration: int = 5000
        ):

    training_steps_made = 0
    evaluations_results = []
    while training_steps_made < simulation_duration:

        # Evaluate if we need to
        if training_steps_made >= len(evaluations_results) * steps_between_two_eval:
            evaluations_results.append(
                (training_steps_made,
                evaluation(agent, env, tests_duration=steps_per_episode, nb_tests=10))
            )

        # Continue training
        duration = min(steps_per_episode, simulation_duration - training_steps_made)
        training_steps_made += episode(agent, env, test_episode=False, nb_steps=duration)

    # Final evaluation (last point of the curve)
    if training_steps_made >= len(evaluations_results) * steps_between_two_eval:
        evaluations_results.append(
            (training_steps_made,
             evaluation(agent, env, tests_duration=steps_per_episode, nb_tests=10))
        )

    return evaluations_results


