# RLF
Reinforcement Learning Framework

### Disclaimer

This project is still a work in progress and some stuff might not work.
For example the Dockerfile have not been tested.

## Give it a try
run:
```PYTHONPATH=$PTHONPATH:. python3 experiments/goal_condtioned_rl/main.py```

## Installation

 - **Fork** and **clone** the repository.

### Using a venv

```bash
python3 -m venv venv
. venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Using the dockerfile

```bash
docker build . --tag rlf:latest
docker run -v "$(pwd):/rlf/" --gpus all rlf:latest python3 /rlf/experiments/goal_condtioned_rl/main.py
```
`--gpus all` allows you to run it using cuda.
`-v "$(pwd):/rlf/"` mount the repository as "/rlf/" so it can be used by the docker container.

## Usage

 - Create your own project with a copy (download using wget) or a fork of this project.
 - Add an experiment directory in <your-project>/experiments and put your code inside.
 - Use the agents and environments you got from this repo in this directory (or other ones from gym ...).
 - You can start with a copy of <your-project>/experiments/goal_conditioned_rl/ which is an example.

## /experiments/goal_conditioned_rl/ example

#### Parrallel simulations
The directory `/experiments/goal_conditioned_rl/` contains a `main.py` file that launch simulations in parrallel from `simulation.py`, which launch a goal-conditioned simulation between an agent and an environment.

#### Extract and plot results
During the simulation, it stores its outputs in  experiments/goal_conditioned_rl/outputs/<environment_name>/<agent_name>. Then `plot_results.py` read these outputs and generate a goal-reaching accuracy plot from them.

#### Save and reload
The `simulation.py` saves the agent and the ongoing simulation after each evaluation. So if your computer crash, you can re-launch the `main.py`, and the `simulation.py` will load the last simulation from `outputs/` directory and continue it, even if you have a lot of simulations running in parrallel.

#### Discord notification
Once a simulation is done, if you put a discord webhook in `simulation.py`, it will send a message on discord channel associated with the webhook (you don't need to have a server for this, just create a discord server, and associate a channel with a webhook).
The `send_discord_message` function can be imported from rlf/utils so it's not a big deal to use it in your own experiments.

## Contribute

Feel free to open an issue on any subject and any question.
The code in `agents/`, `environments/` and `utils/` aim to be implemented in order to be as much **easy to use** as possiple. If you have any idea about how to improve it, feel free to open an issue or to contact me, I will be glad to discuss about it and modify the code in consequence.

