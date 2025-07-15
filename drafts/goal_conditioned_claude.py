import pathlib
from typing import Tuple, List
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import time
import threading
from collections import defaultdict
import copy

from rlnav import PointMazeV0
from sciborg import ConditionedAgent, ConditioningWrapper, SAC, HER, DDPG, TD3
from statistics import mean

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
warnings.filterwarnings(
    "ignore",
    message="No artists with labels found to put in legend.*",
    category=UserWarning
)

grid_size = 11
start_x = grid_size // 2
maze_array = [[1] * grid_size]
for _ in range(grid_size - 2):
    maze_array.append([1] + [0] * (grid_size - 2) + [1])
maze_array.append([1] * grid_size)
maze_array[start_x][start_x] = 2  # Start_tile

environment = PointMazeV0(maze_array=maze_array, goal_conditioned=True, action_noise=0.0, sparse_reward=True)

agents = [
    # {'class': ConditioningWrapper, "args": (SAC, environment.observation_space, environment.action_space, environment.goal_space), "kwargs": {}},
    # {'class': ConditioningWrapper, "args": (DDPG, environment.observation_space, environment.action_space, environment.goal_space), "kwargs": {}},
    # {'class': ConditioningWrapper, "args": (TD3, environment.observation_space, environment.action_space, environment.goal_space), "kwargs": {}},
    {'class': HER, "args": (SAC, environment.observation_space, environment.action_space, environment.goal_space), "kwargs": {}},
    {'class': HER, "args": (DDPG, environment.observation_space, environment.action_space, environment.goal_space), "kwargs": {}},
    {'class': HER, "args": (TD3, environment.observation_space, environment.action_space, environment.goal_space), "kwargs": {}}
]


def episode(
        agent: ConditionedAgent,
        env,
        test_episode: bool = False,
        nb_steps: int = 70, ) -> Tuple[int, bool]:
    observation, info = env.reset()
    goal = info["goal"]
    agent.start_episode(observation, goal, test_episode=test_episode)

    result = False
    terminated = False
    steps_made = 0
    while not terminated:
        action = agent.action(observation)
        observation, reward, terminated, _, info = env.step(action)
        agent.process_interaction(action, reward, observation, terminated)
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


def run_single_simulation(agent_config, simulation_id, results_queue,
                          steps_per_episode=70, steps_between_two_eval=500,
                          simulation_duration=5000):
    """Run a single simulation for one agent and put results in queue"""

    # Create fresh environment for this process
    env = PointMazeV0(maze_array=maze_array, goal_conditioned=True,
                      action_noise=0.0, sparse_reward=True)

    # Create agent instance
    agent = agent_config["class"](*agent_config["args"], **agent_config["kwargs"])

    # Run simulation
    results = simulation(
        agent, env,
        steps_per_episode=steps_per_episode,
        steps_between_two_eval=steps_between_two_eval,
        simulation_duration=simulation_duration
    )

    # Put results in queue
    results_queue.put((agent.name, simulation_id, results))


def average_evaluation_results(all_results):
    """
    Average evaluation results across simulations for each agent.
    Handles different numbers of evaluations per simulation.
    """
    averaged_results = {}

    for agent_name, simulations in all_results.items():
        if not simulations:
            continue

        # Find maximum number of evaluations across all simulations
        max_evals = max(len(sim_results) for sim_results in simulations if sim_results)

        averaged_steps = []
        averaged_accuracies = []

        for eval_idx in range(max_evals):
            steps_at_eval = []
            accuracies_at_eval = []

            for sim_results in simulations:
                if sim_results and eval_idx < len(sim_results):
                    steps_at_eval.append(sim_results[eval_idx][0])
                    accuracies_at_eval.append(sim_results[eval_idx][1])

            if accuracies_at_eval:  # Only add if we have data
                averaged_steps.append(np.mean(steps_at_eval))
                averaged_accuracies.append(np.mean(accuracies_at_eval))

        averaged_results[agent_name] = (averaged_steps, averaged_accuracies)

    return averaged_results


def update_plot(all_results, agent_names, colors):
    """Update the plot with current results"""
    plt.clf()

    averaged_results = average_evaluation_results(all_results)

    for i, agent_name in enumerate(agent_names):
        if agent_name in averaged_results:
            steps, accuracies = averaged_results[agent_name]
            if steps and accuracies:
                plt.plot(steps, accuracies,
                         color=colors[i % len(colors)],
                         label=agent_name,
                         marker='o',
                         markersize=4)

    plt.xlabel('Training Steps')
    plt.ylabel('Success Rate')
    plt.title('Agent Performance Comparison (Live Update)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.pause(0.1)


def main(n_simulations=2,
         update_interval=10,
         steps_per_episode=70,
         steps_between_two_eval=500,
         simulation_duration=5000,
         results: dict = None):
    """
    Main function to run parallel simulations and update plots

    Args:
        n_simulations: Number of simulations to run for each agent
        update_interval: Time in seconds between plot updates
        steps_per_episode: Steps per episode in simulation
        steps_between_two_eval: Steps between evaluations
        simulation_duration: Total training steps per simulation
    """
    if results is None:
        results = defaultdict(list)
    agent_names = [agent["class"](*agent["args"], **agent["kwargs"]).name for agent in agents]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    results_queue = mp.Queue()
    processes = []

    # Storage for all results
    completed_simulations = defaultdict(int)

    # Start all simulations
    print(
        f"Starting {len(agents)} agents × {n_simulations} simulations = {len(agents) * n_simulations} total simulations")

    for agent in agents:
        for sim_id in range(n_simulations):
            process = mp.Process(
                target=run_single_simulation,
                args=(agent, sim_id, results_queue,
                      steps_per_episode, steps_between_two_eval, simulation_duration)
            )
            process.start()
            processes.append(process)

    # Setup live plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Monitor results and update plot
    total_expected = len(agents) * n_simulations
    completed = 0

    while completed < total_expected:
        # Check for new results
        while not results_queue.empty():
            try:
                agent_name, sim_id, simulation_results = results_queue.get_nowait()
                if simulation_results is not None:
                    results[agent_name].append(simulation_results)
                    completed_simulations[agent_name] += 1
                    print(f"Completed: {agent_name} simulation {sim_id + 1}")
                completed += 1
            except:
                break

        # Update plot
        update_plot(results, agent_names, colors)

        # Print progress
        if completed % 5 == 0 or completed == total_expected:
            print(f"Progress: {completed}/{total_expected} simulations completed")

        time.sleep(update_interval)

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Final plot update
    update_plot(results, agent_names, colors)
    plt.ioff()
    plt.show()

    # Print final summary
    print("\n=== Final Results Summary ===")
    averaged_results = average_evaluation_results(results)
    for agent_name in agent_names:
        if agent_name in averaged_results:
            steps, accuracies = averaged_results[agent_name]
            if accuracies:
                final_accuracy = accuracies[-1]
                print(f"{agent_name}: Final accuracy = {final_accuracy:.3f}")

    return results


if __name__ == "__main__":
    # Run with default parameters
    import pickle

    if pathlib.Path('simulation_results.pkl').exists():
        results = pickle.load(open('simulation_results.pkl', 'rb'))
    else:
        results = defaultdict(list)

    results = main(
        n_simulations=2,  # Number of simulations per agent
        update_interval=10,  # Plot update interval in seconds
        steps_per_episode=70,  # Steps per episode
        steps_between_two_eval=300,  # Steps between evaluations
        simulation_duration=5000,  # Total training steps
        results=results
    )

    # Optionally save results
    pickle.dump(results, open('simulation_results.pkl', 'wb'))