"""
In this experiment, we compare many agents against each others in a goal-conditioned grid world.

"""
import logging
import sys
import pathlib
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
from environments import GoalConditionedGridWorld, GridWorldMapsIndex
from agents import DQN, DistributionalDQN, GoalConditionedWrapper, HER, TILO
from simulation import simulation
from utils import send_discord_message, ProgressBar, print_replace_above
import time


path_to_rlf = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(path_to_rlf))

discord_webhook_url = ""


def simulation_status_notifier(p_info):
    done = False
    first = True
    progress_bars = []
    for i in p_info:
        progress_bars.append(ProgressBar(name=i['description'], max_value=i['nb_training_steps'],
                                         print_length=160, name_print_length=30))

    while not done:
        done = True
        try:
            for index, (bar, i) in enumerate(zip(progress_bars, p_info)):
                # Set done to False if one process is not
                if not i["done"]:
                    done = False
                bar.n = min(i["training_steps_done"], i["nb_training_steps"])
                print_replace_above(0 if first else len(p_info) - index, bar)
            time.sleep(2)
            first = False
        except FileNotFoundError as e:
            print("ERROR. file not found " + str(e.filename))
            return
        except Exception as e:
            logging.exception(e)  # Don't seem to be a problem.
    print("ALL DONE.")


if __name__ == "__main__":
    environment = GoalConditionedGridWorld(map_name=GridWorldMapsIndex.EMPTY.value)

    if discord_webhook_url == "":
        print("Add a discord webhook url if you want to receive discord messages once the simulations are done.")

    agents = [
        GoalConditionedWrapper(DQN, environment.observation_space, environment.action_space,
                               environment.get_goal_from_observation),
        HER(DQN, environment.observation_space, environment.action_space, environment.get_goal_from_observation),
        TILO(DQN, environment.observation_space, environment.action_space, environment.get_goal_from_observation),
        GoalConditionedWrapper(DistributionalDQN, environment.observation_space, environment.action_space,
                               environment.get_goal_from_observation),
        HER(DistributionalDQN, environment.observation_space, environment.action_space,
            environment.get_goal_from_observation),
        TILO(DistributionalDQN, environment.observation_space, environment.action_space,
             environment.get_goal_from_observation)
    ]

    torch.multiprocessing.set_start_method('spawn')

    simulations = []
    manager = Manager()
    simulations_information = manager.list()
    nb_simulation_per_agent = 3
    for agent in agents:
        for simulation_id in range(nb_simulation_per_agent):
            nb_training_steps = 30000

            # Setup information that will be shared with the current process
            # It is used to show the advancement of each simulation.
            info = manager.dict()
            info['simulation_id'] = simulation_id
            info['description'] = agent.name + " - sim. " + str(simulation_id)
            info['training_steps_done'] = 0
            info['done'] = False
            info['nb_training_steps'] = nb_training_steps
            simulations_information.append(info)

            # Setup and launch simulation
            p = mp.Process(target=simulation,
                           args=(environment.copy(), agent.copy(), info),
                           kwargs={"nb_training_steps": nb_training_steps,
                                   "episodes_max_duration": 100,
                                   "nb_steps_before_evaluation": 2000,
                                   "nb_tests_per_evaluation": 20,
                                   "simulation_id": simulation_id,
                                   "device": "cuda",
                                   "save": True,
                                   "load": True,
                                   "verbose": False
                                   })

            simulations.append(p)
            p.start()
    notifier = mp.Process(target=simulation_status_notifier, args=(simulations_information,))
    notifier.start()
    print("Simulations launched")

    for simu, info in zip(simulations, simulations_information):
        simu.join()
        if discord_webhook_url != "":
            send_discord_message("[DONE] " + "hbrl simulation " + info["description"],
                                 discord_webhook_url)
        info["done"] = True
