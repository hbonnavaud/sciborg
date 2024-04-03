# RLF
Reinforcement Learning Framework

### Disclaimer

This project is still a work in progress.
Everything here should work, but expect a high frequency of modifications and "improvements" over time.

#### TODOs (in this order)

 - Add a basic RL minimal example on cartpole (similar to the one in goal-conditioned RL).
 - Create a wiki, because this readme is becoming messy ... It should be precise enough to be used as a documentation.
 - clean-up environments code (should not change their behaviour)
 - Create new agents:
   - DIAYN: an agent that wrap an value-based RL agent and make it learn skills
   - A model-based agent (probably MuZero if I succeed ...)
   - ICM (intrinsic curiosity module)
 - Propose a new usage with a `rlf` bash command. \
   The current easier usage, is to fork or download this repository, which force you to clone/download every
   agent, environment, or experiment in this repository. 
   It is not a big deal now but it can become one in the future as new agents and elements are added. \
   The command will have a simple structure:
   - `rlf init <project_name>`: create a repository named `<project_name>` with the basic structure of this repository. 
   - `rlf use <component>`: search for the given component (could be an agent, an environment, en experiment folder)
      on the main branch, and download it with all used classes. \
      Ex: `rlf use DQN` download classes "DQN", "ValueBasedAgent", and "Agent", and put them in the agent folder.
   - `rlf update <component>` update a component if it has been modified on the git's main branch.

## Give it a try
run:
```PYTHONPATH=$PTHONPATH:. python3 experiments/goal_condtioned_rl/main.py```

NB: run it from a terminal, it uses ANSI characters and will not be printed well if you run it from py-charm.

## Installation

 - **Fork** and **clone** the repository.
 - or download it with `wget`: 
```
PROJECT_NAME="your_new_rl_project" && \
    wget https://github.com/hbonnavaud/RLFramework/archive/master.zip && \
    unzip master.zip && \
    mv RLFramework-main $PROJECT_NAME && \
    rm master.zip
```

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
 - Import the agents and environments you need in your experiment directory (or other ones from gym ...).
 - You can start with a copy of <your-project>/experiments/goal_conditioned_rl/ which is an example (explained bellow).

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

## Agents

< explanations incoming >

## Environments

< explanations incoming >

## Contribute

Feel free to open an issue on any subject and any question.
The code in `agents/`, `environments/` and `utils/` aim to be implemented in order to be as much **easy to use** as possiple. If you have any idea about how to improve it, feel free to open an issue or to contact me, I will be glad to discuss about it and modify the code in consequence.

PRs for new agents and new environments are highly welcome (Fork + pull request only!).
