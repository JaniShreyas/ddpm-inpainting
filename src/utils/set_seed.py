import torch
import numpy as np
import random

def set_seed(seed: int):
    if seed is None:
        print("No seed provided. Skipping seed setting")
        return
    
    print(f"Setting global seed to: {seed}")

    # In built random
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)