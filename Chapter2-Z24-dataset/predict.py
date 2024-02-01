import numpy as np

# Rule: 1 to 8, or 0 to 7 are healthy cases
HEALTHY_STATES = np.arange(8)

def binarize(y):
    return np.array([0 if x in HEALTHY_STATES else 1 for x in y])
