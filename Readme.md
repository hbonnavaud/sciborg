# HBRL

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

### Using a docker

```bash
docker build . --tag rlf:latest
docker run --gpus all 
```
TODO: incoming
