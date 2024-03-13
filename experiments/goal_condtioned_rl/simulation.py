import json
import os.path
import pickle
import sys
import time
import matplotlib.pyplot as plt
import torch
from agents import Agent, GoalConditionedAgent
from agents.value_based_agents.dqn import DQN
from utils import ProgressBar, Ansi, print_replace_above, create_dir, empty_dir, send_discord_message
import pprint

"""
This experiment script make an agent interact with an environment.
It evaluate the agent, save it after each evaluation, and print the evolution of the process.
This file work with classic value-based agent and their goal-conditioned version. 
"""


def print_simulation_status(agent, environment, training_progressbar, evaluation_progressbar, under_evaluation=False,
                            last_save_at_step=None, nb_evaluation_done=None, replace_previous_one=False):
    message_width = 100
    simulation_name = "Training " + agent.name + " in " + environment.name
    header_fill_width = (message_width - 4 - len(simulation_name)) // 2
    message_width = header_fill_width * 2 + len(simulation_name) + 4

    # Print message box top
    print_replace_above(7 if replace_previous_one else 0, "")
    print_replace_above(6 if replace_previous_one else 0, "")
    print_replace_above(5 if replace_previous_one else 0,
                        Ansi.box_top_left + Ansi.box_horizontal_bar * header_fill_width + " " + simulation_name + " "
                        + Ansi.box_horizontal_bar * header_fill_width + Ansi.box_top_right)

    # Print training status bar
    print_replace_above(4 if replace_previous_one else 0,
                        Ansi.box_vertical_bar + " " + str(training_progressbar) + " " + Ansi.box_vertical_bar)

    # Print evaluation status bar
    if under_evaluation:
        print_replace_above(3 if replace_previous_one else 0,
                            Ansi.box_vertical_bar + " " + str(evaluation_progressbar) + " " + Ansi.box_vertical_bar)
    else:
        msg = str(nb_evaluation_done) + " evaluations done."
        print_replace_above(3 if replace_previous_one else 0, Ansi.box_vertical_bar + " "
                            + msg + " " * (message_width - len(msg) - 3) + Ansi.box_vertical_bar)

    # Print save message
    if last_save_at_step is None:
        msg = "Not saved yet."
    elif last_save_at_step == -1:
        msg = "Save is disable, agent will not be saved."
    else:
        msg = ("Agent saved at step " + str(last_save_at_step)
               + " (at " + str(last_save_at_step // training_progressbar.max_value) + "%)")
    print_replace_above(2 if replace_previous_one else 0,
                        Ansi.box_vertical_bar + " " + msg + " " * (message_width - len(msg) - 3) + Ansi.box_vertical_bar)

    # Print message box's bottom
    print_replace_above(1 if replace_previous_one else 0,
                        Ansi.box_bottom_left + Ansi.box_horizontal_bar * (message_width - 2) + Ansi.box_bottom_right)


def simulation(environment, agent: Agent, shared_info, simulation_id=0, nb_training_steps=500,
               episodes_max_duration=200, nb_tests_per_evaluation=20, nb_steps_before_evaluation=50, plot=False,
               verbose=True, generate_videos=False, device="cuda", save_agent=True):

    simulation_output_directory = os.path.dirname(os.path.abspath(__file__))
    simulation_output_directory = os.path.join(simulation_output_directory, "outputs", environment.__class__.__name__,
                                               agent.name.replace(" ", "_"), "simulation_" + str(simulation_id))

    create_dir(simulation_output_directory)
    sys.stdout = open(os.path.join(simulation_output_directory, "stdout.txt"), "w")

    simulation_information = {"environment": environment.name, "agent": agent.name, "simulation_id": simulation_id,
                              "nb_training_steps": nb_training_steps, "episodes_max_duration": episodes_max_duration,
                              "nb_tests_per_evaluation": nb_tests_per_evaluation,
                              "nb_steps_before_evaluation": nb_steps_before_evaluation, "plot": plot,
                              "verbose": verbose, "generate_videos": generate_videos, "device": device
                              }

    last_save_at_step = None if save_agent else -1

    if os.path.isdir(os.path.join(simulation_output_directory, "agent")):
        # Verify if the simulation is indeed the same
        sav_simu_info = json.load(open(os.path.join(simulation_output_directory, "simulation_infos.json"), "r"))
        for k, v in sav_simu_info.items():
            assert k in simulation_information and simulation_information[k] == v, \
                "Loading error, saved simulation info is different"

        agent.load(os.path.join(simulation_output_directory, "agent"))

        # Load the saved simulation
        with open(os.path.join(simulation_output_directory, "simulation_metrics.pkl"), "rb") as f:
            simulation_metrics = pickle.load(f)
            shared_info["training_steps_done"] = simulation_metrics["training_steps_done"]
            evaluations_results = simulation_metrics["evaluations_results"]
            evaluation_time_steps = simulation_metrics["evaluation_time_steps"]
        del f, simulation_metrics

    else:
        empty_dir(simulation_output_directory, send_to_trash=False)
        sys.stdout = open(os.path.join(simulation_output_directory, "stdout.txt"), "w")

        print("############################################################################")
        print("Launching simulation with informations:")
        print()
        pprint.pprint(simulation_information)
        print()
        print("############################################################################")
        shared_info["training_steps_done"] = 0
        evaluations_results = []
        evaluation_time_steps = []

    if verbose:  # PRINT MESSAGE BOX
        message_width = 100
        simulation_name = "Training " + agent.name + " in " + environment.name
        header_fill_width = (message_width - 4 - len(simulation_name)) // 2
        message_width = header_fill_width * 2 + len(simulation_name) + 4

        training_progressbar = ProgressBar("Training", nb_training_steps, print_length=message_width - 4)
        evaluation_progressbar = ProgressBar("Evaluation", nb_tests_per_evaluation, print_length=message_width - 4)

        print_simulation_status(agent, environment, training_progressbar, evaluation_progressbar, nb_evaluation_done=0,
                                last_save_at_step=last_save_at_step)

    agent.set_device(torch.device(device))

    def save():
        nonlocal last_save_at_step
        # Save
        agent.save(os.path.join(simulation_output_directory, 'agent'))
        with open(os.path.join(simulation_output_directory, "simulation_metrics.pkl"), "wb") as f:
            pickle.dump({
                "training_steps_done": shared_info["training_steps_done"],
                "evaluations_results": evaluations_results,
                "evaluation_time_steps": evaluation_time_steps
            }, f)
        json.dump(simulation_information, open(os.path.join(simulation_output_directory, "simulation_infos.json"), "w"))

        if verbose:
            last_save_at_step = shared_info["training_steps_done"]
            print_simulation_status(agent, environment, training_progressbar, evaluation_progressbar,
                                    nb_evaluation_done=len(evaluations_results),
                                    last_save_at_step=last_save_at_step)

    def episode(environment_, agent_, test=False):
        assert hasattr(environment_, "goal_space")
        assert isinstance(agent_, GoalConditionedAgent)

        observation, goal = environment_.reset()
        # test_episode = True disable exploration from the agent (<=> force exploitation) and disable learning.
        agent_.start_episode(observation, goal, test_episode=test)

        score = 0
        steps_done = 0
        for step_id in range(episodes_max_duration):

            if not test and evaluation_time_steps == len(evaluation_time_steps) * nb_steps_before_evaluation:
                evaluation()

            action = agent_.action(observation)
            observation, reward, done, info = environment_.step(action)
            agent_.process_interaction(action, reward, observation, done)
            steps_done += 1
            if not test:
                shared_info["training_steps_done"] += 1

            if info["reached"]:
                score = 1
                break
            if done:
                break
        agent_.stop_episode()
        return score, steps_done

    def evaluation():
        agent_to_eval = agent.copy()
        evaluation_environment = environment.copy()
        score_sum = 0
        for test_id in range(nb_tests_per_evaluation):
            score_sum += episode(evaluation_environment, agent_to_eval, test=True)[0]

            if verbose:
                evaluation_progressbar.step(1)
                print_simulation_status(agent, environment, training_progressbar, evaluation_progressbar,
                                        under_evaluation=True, replace_previous_one=True,
                                        last_save_at_step=last_save_at_step)
                time.sleep(0.1)

        if verbose:
            evaluation_progressbar.reset()
            print_simulation_status(agent, environment, training_progressbar, evaluation_progressbar,
                                    nb_evaluation_done=len(evaluations_results) + 1, replace_previous_one=True,
                                    last_save_at_step=last_save_at_step)

        evaluations_results.append(score_sum / nb_tests_per_evaluation)
        evaluation_time_steps.append(shared_info["training_steps_done"])
        save()

    pbar_incremented = 0

    # training
    while shared_info["training_steps_done"] < nb_training_steps:
        _, episode_duration = episode(environment, agent, test=False)

        if evaluation_time_steps == len(evaluation_time_steps) * nb_steps_before_evaluation:
            evaluation()

        if verbose:
            training_progressbar.step(episode_duration)
            pbar_incremented += episode_duration
            print_simulation_status(agent, environment, training_progressbar, evaluation_progressbar,
                                    nb_evaluation_done=len(evaluations_results), replace_previous_one=True,
                                    last_save_at_step=last_save_at_step)

    save()
