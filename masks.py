import tensorflow as tf
import numpy as np

def create_padding_mask(decoder_token_ids):
    '''
    Creates a matrix mask for the padding cells
    
    Arguments:
        decoder_token_ids -- (n, m) matrix
    
    Returns:
        mask -- (n, 1, m) binary tensor
    '''

    mask = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)

    return mask[:, np.newaxis, :]

def create_look_ahead_mask(sequence_length):
    '''
    Returns a lower triangular matrix filled with ones. Used during training to mask the output seq
    '''

    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask
