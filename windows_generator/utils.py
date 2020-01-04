
import numpy as np

def shuffle_equally(*args):
    indices = np.random.permutation(len(args[-1]), )
    return [
        x[indices]
        for x in args
    ]
    