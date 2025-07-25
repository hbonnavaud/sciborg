from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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
    eval_agent = agent.copy()
    nb_successes = 0
    for test_id in range(nb_tests):
        if episode(eval_agent, env, test_episode=True, nb_steps=tests_duration):
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


def get_agent_name(agent):
    """Get a readable name for the agent"""
    if hasattr(agent, 'algorithm'):
        # For HER agents
        return f"HER({agent.algorithm.__name__})"
    elif hasattr(agent, 'wrapped_class'):
        # For ConditioningWrapper agents
        return f"Cond({agent.wrapped_class.__name__})"
    else:
        return str(type(agent).__name__)


def average_evaluation_results(all_results):
    """
    Average evaluation results across multiple simulations.
    Each simulation can have different numbers of evaluation points.
    """
    if not all_results:
        return [], []

    # Group results by evaluation step
    step_to_values = defaultdict(list)

    for sim_results in all_results:
        for step, accuracy in sim_results:
            step_to_values[step].append(accuracy)

    # Sort by steps and compute averages
    sorted_steps = sorted(step_to_values.keys())
    avg_steps = []
    avg_accuracies = []

    for step in sorted_steps:
        avg_steps.append(step)
        avg_accuracies.append(mean(step_to_values[step]))

    return avg_steps, avg_accuracies


def main(n_simulations=3, simulation_duration=5000, steps_between_eval=500):
    """
    Run multiple simulations for each agent and plot averaged results.

    Args:
        n_simulations: Number of simulations per agent
        simulation_duration: Total training steps per simulation
        steps_between_eval: Steps between evaluations
    """

    # Store results for all agents
    all_agent_results = {}

    # Run simulations for each agent
    for i, agent in enumerate(agents):
        agent_name = get_agent_name(agent)
        print(f"Running {n_simulations} simulations for {agent_name}...")

        # Store results for this agent across all simulations
        agent_simulations = []

        for sim_idx in range(n_simulations):
            print(f"  Simulation {sim_idx + 1}/{n_simulations}")

            # Create a fresh environment for each simulation
            env = PointMazeV0(maze_array=maze_array, goal_conditioned=True,
                              action_noise=0.0, sparse_reward=True)

            # Run simulation
            results = simulation(
                agent=agent,
                env=env,
                steps_per_episode=70,
                steps_between_two_eval=steps_between_eval,
                simulation_duration=simulation_duration
            )

            agent_simulations.append(results)

        # Average results across simulations
        avg_steps, avg_accuracies = average_evaluation_results(agent_simulations)
        all_agent_results[agent_name] = (avg_steps, avg_accuracies)

        print(f"  Completed {agent_name}: Final average accuracy = {avg_accuracies[-1]:.3f}")

    # Plot results
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, (agent_name, (steps, accuracies)) in enumerate(all_agent_results.items()):
        plt.plot(steps, accuracies,
                 label=f'{agent_name} (n={n_simulations})',
                 color=colors[i % len(colors)],
                 linewidth=2,
                 marker='o',
                 markersize=4)

    plt.xlabel('Training Steps')
    plt.ylabel('Success Rate')
    plt.title(f'Agent Performance Comparison (Averaged over {n_simulations} simulations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, simulation_duration)
    plt.ylim(0, 1.0)

    # Add some statistics
    plt.tight_layout()
    plt.show()

    # Print final results summary
    print("\nFinal Results Summary:")
    print("-" * 50)
    for agent_name, (steps, accuracies) in all_agent_results.items():
        print(f"{agent_name:15s}: {accuracies[-1]:.3f} (final success rate)")

    return all_agent_results


if __name__ == "__main__":
    # Run the main function with default parameters
    # You can adjust these parameters as needed
    results = main(n_simulations=3, simulation_duration=5000, steps_between_eval=500)