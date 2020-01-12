
import numpy as np


def one_hot_encode(string):
    string = string.lower()
    matrix = np.eye(4)
    return np.array(matrix[
        [
            "actg".find(c)
            for c in string
        ]
    ])


def one_hot_encoder(sequences):
    encoded = np.array([
        one_hot_encode(sequence)
        for sequence in sequences
    ])
    return encoded, encoded
