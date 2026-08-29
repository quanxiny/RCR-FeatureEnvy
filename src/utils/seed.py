import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fix every random source used by the stable baseline."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def capture_rng_state(sampling_rng: random.Random) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "sampling": sampling_rng.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, sampling_rng: random.Random) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints are commonly loaded with map_location set to the training
    # device, which also moves the serialized RNG ByteTensors. PyTorch's RNG
    # restoration APIs require these state tensors to reside on the CPU.
    torch.set_rng_state(state["torch"].cpu())
    sampling_rng.setstate(state["sampling"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
