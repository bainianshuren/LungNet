# utils/common.py
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set all random seeds for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_seeds(seeds=[42, 123, 777]):
    """Iterate over multiple seeds for repeated experiments."""
    for seed in seeds:
        set_seed(seed)
        yield seed