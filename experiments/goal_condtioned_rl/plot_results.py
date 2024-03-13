import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


current_dir = os.path.dirname(os.path.realpath(__file__))
outputs_dir = os.path.join(current_dir, "outputs", "GoalConditionedGridWorld")

print()
print()
print("THIS DO NOT WORK YET, IT IS A WORK IN PROGRESS")
print()
print()

for agent in os.listdir(outputs_dir):
    agent_path = os.path.join(outputs_dir, agent)
    eval_ts_ref = None
    all_agent_evaluation_results = []
    for simulation in os.listdir(agent_path):
        simulation_path = os.path.join(agent_path, simulation)

        try:
            simulation_metrics = os.path.join(simulation_path, "simulation_metrics.pkl")
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

    all_agent_evaluation_results = np.array(all_agent_evaluation_results)
    results_means = np.mean(all_agent_evaluation_results, axis=0)
    results_stds = np.std(all_agent_evaluation_results, axis=0)

    plt.plot(eval_ts_ref, results_means, label=agent)
    plt.fill_between(eval_ts_ref, results_means + results_stds, results_means - results_stds, alpha=0.3)

plt.xlabel("training interactions")
plt.ylabel("goal-reaching accuracy")
plt.legend()
plt.show()
