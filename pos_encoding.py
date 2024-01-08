import numpy as np
import tensorflow as tf

def get_angles(positions, k, d):
    '''
        positions: Column vector containing positions [[0], [1], [N-1]]
        k: Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
        d: Dimention size

        returns:
            angles: shape (N, d)
    '''

    i = np.ceil(k / 2)
    angles = positions / (10000 ** (2*i / d))

    return angles


def positional_encoding(max_seq_length, encoding_dim):
    '''
        max_seq_length: int
        encoding_dim: int

        Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    '''

    positions = np.arange(max_seq_length)[:, np.newaxis]
    k = np.arange(encoding_dim)[np.newaxis, :]

    angles = get_angles(positions, k, encoding_dim)

    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    pos_encodings = angles[np.newaxis, ...]

    return tf.cast(pos_encodings, dtype=tf.float32)