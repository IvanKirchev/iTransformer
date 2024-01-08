import tensorflow as tf
from utils import FullyConnected

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, embedding_dim, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        ) 
        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        ) 

        self.ffn = FullyConnected(embedding_dim=embedding_dim, fully_connected_dim=fully_connected_dim)

        self.add_norm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.add_norm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.add_norm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, encoder_out, training, look_ahead_mask, padding_mask):
        
        mult_attn_out1, attn_weights_block1  = self.mha1(
            query = x, key = x, value = x, 
            attention_mask = look_ahead_mask, 
            training = training, return_attention_scores=True
        )

        Q1 = self.add_norm1(tf.add(x, mult_attn_out1))


        mult_attn_out2, attn_weights_block2 = self.mha2(
            query = Q1, key = encoder_out, value = encoder_out, 
            attention_mask = padding_mask,  
            training = training, return_attention_scores=True
        )

        mult_attn_out2 = self.add_norm2(tf.add(Q1, mult_attn_out2))

        ffn_out = self.ffn(mult_attn_out2)
        ffn_output = self.dropout_ffn(ffn_output, training = training)

        out = self.add_norm3(tf.add(mult_attn_out2, ffn_output))

        return out, mult_attn_out1, mult_attn_out2
