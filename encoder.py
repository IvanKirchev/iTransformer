import tensorflow as tf
from pos_encoding import positional_encoding
from utils import FullyConnected
import numpy as np

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        
        super(EncoderLayer, self).__init__()
        
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embedding_dim,
            dropout=dropout_rate
        )

        self.ffn = FullyConnected(
            embedding_dim=embedding_dim, 
            fully_connected_dim=fully_connected_dim
        )

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        '''
        Forward pass for the Encoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        '''

        multihead = self.mha(query=x, value=x, key=x, training=training, attention_mask=mask)
        print(self.mha.weights)

        add_norm1 = self.norm1(x + multihead)

        ffn_output = self.ffn(add_norm1)

        ffn_output = self.dropout_ffn(ffn_output)

        encoder_layer_output = self.norm2(ffn_output + add_norm1)

        return encoder_layer_output
    
class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size, output_dim=self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)

        self.enc_layers = [EncoderLayer(self.embedding_dim, num_heads, 
                                        fully_connected_dim, dropout_rate, 
                                        layernorm_eps) 

        for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        # Optional scale down the embedding values
        # Determine during evaluation if this improves anything

        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training = training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x
    



# tf.random.set_seed(10)

# q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
# encoder_layer1 = EncoderLayer(4, 2, 8)

# encoded = encoder_layer1(q, True, np.array([[1, 0, 1]]))