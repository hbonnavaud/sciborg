import json
import pathlib
import pickle
import sys
from statistics import mean

import torch
from agents import Agent, GoalConditionedAgent
from utils import create_dir, empty_dir
import pprint

"""
This experiment script make an agent interact with an environment.
It evaluate the agent, save it after each evaluation, and print the evolution of the process.
This file work with classic value-based agent and their goal-conditioned version. 
"""


def simulation(environment, agent: Agent, shared_info, simulation_id=0, nb_training_steps=500,
               episodes_max_duration=200, nb_tests_per_evaluation=20, nb_steps_before_evaluation=50, plot=False,
               verbose=True, generate_videos=False, device="cuda", save=True, load=True, name=""):
    if name == "":
        name = agent.name.replace(" ", "_")
    simulation_output_directory = (pathlib.Path(__file__).parent / "outputs"
                                   / environment.name / name / ("simu_" + str(simulation_id)))
    create_dir(simulation_output_directory)
    sys.stdout = open(simulation_output_directory / "stdout.txt", "w")

    simulation_information = {"environment": environment.name, "agent": agent.name, "simulation_id": simulation_id,
                              "nb_training_steps": nb_training_steps, "episodes_max_duration": episodes_max_duration,
                              "nb_tests_per_evaluation": nb_tests_per_evaluation,
                              "nb_steps_before_evaluation": nb_steps_before_evaluation, "plot": plot,
                              "verbose": verbose, "generate_videos": generate_videos, "device": device
                              }

    ##############################################
    # Check saved simulations and load if needed #
    ##############################################
    # Should we load a previously stopped launch of this simulation
    if load and (simulation_output_directory / "agent").is_dir():
        # Verify if the simulation is indeed the same
        sav_simu_info = json.load(open(simulation_output_directory / "simulation_infos.json", "r"))

        for k, v in sav_simu_info.items():
            assert k in simulation_information and simulation_information[k] == v, \
                "Loading error, saved simulation info is different"

        agent.load(simulation_output_directory / "agent")

        # Load the saved simulation
        with open(simulation_output_directory / "simulation_metrics.pkl", "rb") as f:
            simulation_metrics = pickle.load(f)
            shared_info["training_steps_done"] = simulation_metrics["training_steps_done"]
            evaluations_results = simulation_metrics["evaluations_results"]
            evaluation_time_steps = simulation_metrics["evaluation_time_steps"]
        del f, simulation_metrics
    else:
        # There is no saved simulation that match this one, of no loadable simulation.
        empty_dir(simulation_output_directory, send_to_trash=False)
        sys.stdout = open(simulation_output_directory / "stdout.txt", "w")

        print("############################################################################")
        print("Launching simulation with information:")
        print()
        pprint.pprint(simulation_information)
        print()
        print("############################################################################")
        shared_info["training_steps_done"] = 0
        evaluations_results = []
        evaluation_time_steps = []

    agent.set_device(torch.device(device))

    def save():
        # Save
        agent.save(simulation_output_directory / 'agent')
        pickle.dump({
                "training_steps_done": shared_info["training_steps_done"],
                "evaluations_results": evaluations_results,
                "evaluation_time_steps": evaluation_time_steps
            }, open(simulation_output_directory / "simulation_metrics.pkl", "wb"))
        json.dump(simulation_information, open(simulation_output_directory / "simulation_infos.json", "w"))

    def episode(environment_, agent_, test=False):
        assert hasattr(environment_, "goal_space")
        assert isinstance(agent_, GoalConditionedAgent)

        observation, goal = environment_.reset()
        agent_.start_episode(observation, goal, test_episode=test)

        reached = 0
        steps_done = 0
        for step_id in range(episodes_max_duration):

            if not test and shared_info["training_steps_done"] == len(
                    evaluation_time_steps) * nb_steps_before_evaluation:
                evaluation()

            action = agent_.action(observation)
            observation, reward, done, info = environment_.step(action)
            agent_.process_interaction(action, reward, observation, done)
            steps_done += 1
            if not test:
                shared_info["training_steps_done"] += 1

            if info["reached"]:
                reached = 1
                break
            if done:
                break
        agent_.stop_episode()
        return reached, steps_done

    def evaluation():
        agent_to_eval = agent.copy()
        evaluation_environment = environment.copy()
        score_sum = 0
        for test_id in range(nb_tests_per_evaluation):
            result, _ = episode(evaluation_environment, agent_to_eval, test=True)
            score_sum += result
            print("\tEvaluation episode ", test_id, ", result = ", result, sep="")

        evaluations_results.append(score_sum / nb_tests_per_evaluation)
        evaluation_time_steps.append(shared_info["training_steps_done"])
        save()

    # training
    successes = []
    successes_running_average = []
    nb_episodes_made = 0
    while shared_info["training_steps_done"] < nb_training_steps:
        score, episode_duration = episode(environment, agent, test=False)
        successes.append(score)
        print("Training episode", nb_episodes_made, "; result=", score)

        if len(successes) > 20:
            successes_running_average.append(mean(successes[-20:]))

        if shared_info["training_steps_done"] == len(evaluation_time_steps) * nb_steps_before_evaluation:
            evaluation()

    save()
