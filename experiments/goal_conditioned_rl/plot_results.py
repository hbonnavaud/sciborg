import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pathlib


current_dir = pathlib.Path(__file__).parent
outputs_dir = current_dir / "outputs" / "GoalConditionedGridWorld"

plot_one_agent_details = ""

for agent in os.listdir(outputs_dir):
    if plot_one_agent_details != "" and agent != plot_one_agent_details:
        continue
    agent_path = outputs_dir / agent
    eval_ts_ref = None
    all_agent_evaluation_results = []
    for simulation in os.listdir(agent_path):
        simulation_path = agent_path / simulation

        try:
            simulation_metrics = simulation_path / "simulation_metrics.pkl"
            simulation_metrics = pickle.load(open(simulation_metrics, "rb"))
        except:
            print("No outputs found for ", simulation, " of agent ", agent)
            continue
        evaluations_results = simulation_metrics["evaluations_results"]
        evaluation_time_steps = simulation_metrics["evaluation_time_steps"]
        assert len(evaluation_time_steps) == len(evaluations_results)
        if eval_ts_ref is None:
            eval_ts_ref = evaluation_time_steps
        assert len(evaluation_time_steps) == len(eval_ts_ref)
        all_agent_evaluation_results.append(evaluations_results)

    if plot_one_agent_details == "":
        all_agent_evaluation_results = np.array(all_agent_evaluation_results)
        results_means = np.mean(all_agent_evaluation_results, axis=0)
        results_stds = np.std(all_agent_evaluation_results, axis=0)

        plt.plot(eval_ts_ref, results_means, label=agent.replace("_", " "))
        plt.fill_between(eval_ts_ref, results_means + results_stds, results_means - results_stds, alpha=0.3)
    else:
        labels = [plot_one_agent_details] + [None] * (len(all_agent_evaluation_results) - 1)
        plt.plot(eval_ts_ref, np.transpose(np.array(all_agent_evaluation_results)), c="r", label=labels)

plt.xlabel("training interactions")
plt.ylabel("goal-reaching accuracy")
plt.legend()
plt.show()
