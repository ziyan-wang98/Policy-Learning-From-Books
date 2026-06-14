import os
from pathlib import Path


ROOT_PATH = os.environ.get("PLFB_ROOT", str(Path(__file__).resolve().parent))
DATASET_PATH = os.environ.get("PLFB_DATASET_PATH", os.path.join(ROOT_PATH, "data"))
OFFLINE_DATASET_PATH = os.environ.get(
    "PLFB_OFFLINE_DATASET_PATH",
    os.path.join(DATASET_PATH, "offline_dataset-v4"),
)
MODEL_PATH = os.environ.get("PLFB_MODEL_PATH", os.path.join(ROOT_PATH, "saved_model"))
TIZERO_AGENT_PATH = os.environ.get(
    "PLFB_TIZERO_AGENT_PATH",
    os.path.join(ROOT_PATH, "setup", "TiZero", "submission", "tizero_agent"),
)

class KnowledgeType:
    Policy = 'Policy'
    Dynamics = 'Dynamics'
    Reward = 'Reward'


class AlgType:
    CIQL = 'CIQL'
    CQL = 'CQL'
    CQL_REAL = 'CQL_REAL'
    RT = 'RT'
