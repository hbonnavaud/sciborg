from pathlib import Path
from typing import Union
from ..agent import Agent
import pickle


def load_agent(path: Union[Path, str]) -> Agent:
    if isinstance(path, str):
        path = Path(path)
    assert path.is_dir(), "Cannot load agent: path " + str(path) + " not found"

    objects_attributes = pickle.load(open(path / "objects_attributes.pk", "rb"))
    agent_class = objects_attributes["class"]
    agent_init_params = objects_attributes["init_params"]
    return agent_class(**agent_init_params)
