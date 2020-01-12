
import numpy as np

from .one_hot import one_hot_encode
from .decorators import multiprocess


def apply_noise(mask, sequence, n_type):
    y = one_hot_encode(sequence)
    x = np.copy(y)
    if n_type == "uniform":
        x[mask] = [0.25] * 4
    elif n_type == "normal":
        x[mask] = np.random.normal(size=(4,))
    else:
        RuntimeWarning(
            "Unreachable condition, the n_type %s is not valid" % n_type)
    return x, y


@multiprocess
def one_hot_noise(seed, sequences, n_type, mean, cov):
    state = np.random.RandomState()
    state.seed(seed)
    distribution = state.multivariate_normal(
        mean,
        cov,
        size=len(sequences)
    ) > 0.5
    result = np.array([
        apply_noise(mask, sequence, n_type)
        for mask, sequence in zip(distribution, sequences)
    ])
    return result[:, 0], result[:, 1]
