import tensorflow as tf
import numpy as np

def scaled_dot_product_attention(q, k, v, mask):
    '''
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.

        Arguments:
            q -- query shape == (..., seq_len_q, depth)
            k -- key shape == (..., seq_len_k, depth)
            v -- value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable 
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
            output -- attention_weights
    '''

    matmul_qk = tf.matmul(q, k.T) # shape (seq_len_q, seq_len_k)

     # scale matmul_qk
    dk = k.shape[1]
    scaled_attention_logits = matmul_qk / (np.sqrt(dk))


    if mask is not None:
        scaled_attention_logits += ((1 - mask) * -1.0e9)

    attention_weights = tf.keras.activations.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights