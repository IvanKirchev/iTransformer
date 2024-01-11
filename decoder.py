import tensorflow as tf
from utils import FullyConnected
from pos_encoding import positional_encoding
import numpy as np

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

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.embedding_dim)
        self.positional_encoding = positional_encoding(max_seq_length=maximum_position_encoding, encoding_dim=self.embedding_dim)

        self.decoding_layers = [ 
            DecoderLayer(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                fully_connected_dim=fully_connected_dim,
                dropout_rate=dropout_rate,
                layernorm_eps=layernorm_eps
            )
        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, encoder_out, training, 
             look_ahead_mask, padding_mask):
    
        x = self.embedding(x)

        # Scale down
        tf.cast(x, dtype = tf.float32)
        x *= np.sqrt(self.embedding_dim)

        seq_len = tf.shape(x)[1]
        x += self.positional_encoding[:, :seq_len, :]

        x = self.dropout(x, training = training)

        attention_weights = {}
        for i in range(self.decoding_layers):
            x, block1, block2 = self.decoding_layers[i](x, encoder_out, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = block2

        return x, attention_weights