import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def log_line(path, text):
    with open(path, "a") as f:
        f.write(text + "\n")