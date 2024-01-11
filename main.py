from model import Transformer
import tensorflow as tf
import numpy as np
from masks import create_look_ahead_mask, create_padding_mask


if __name__ == "__main__":
    transformer_model = Transformer(
        num_layers=4,
        embedding_dim=300,
        num_heads=6,
        fully_connected_dim=50,
        input_vocab_size=10000,
        target_vocab_size=10000,
        max_positional_encoding_input=300,
        max_positional_encoding_target=300,
    )

    transformer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    
    x = ['My name is Ivan Kirchev', "What is you'r name?", "On how much data are you trained?"]
    y = ['Hello, Ivan. It\'s nice to meet you.', "My name is iTransformer and I'm an AI model", "I'm trained on very little data. It's about 10 sentances"]
    

    batch_size = 32
    source_seq_len = 10
    target_seq_len = 12
    vocab_size = 100

    # Generate random input tensor
    input_data = np.random.randint(0, 10, size=(batch_size, source_seq_len, vocab_size))

    # Generate random output tensor
    output_data = np.random.randint(0, 10, size=(batch_size, target_seq_len, vocab_size))

    def prepare_input_data(input_sentence, output_sentence):
        enc_padding_mask = create_padding_mask(input_sentence)
        dec_padding_mask = create_padding_mask(output_sentence)
        look_ahead_mask = create_look_ahead_mask(output_sentence.shape[1])
        return {'input_sentence': input_sentence, 'output_sentence': output_sentence}, {'enc_padding_mask': enc_padding_mask, 'look_ahead_mask': look_ahead_mask, 'dec_padding_mask': dec_padding_mask}

    dataset = tf.data.Dataset.from_tensor_slices((input_data, output_data))
    dataset = dataset.map(prepare_input_data)

    # TODO: Research how to train the model using model.fit or a custom training loop