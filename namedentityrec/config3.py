from enum import Enum

class DefaultConfig:
    n_epochs = 50
    start_lr = 0.0009

    batch_size = 64
    max_timesteps = 30

    word_embed_size = 12

    # Output sizes of the RNN layers.
    hidden_sizes = 250

    # Character embedding dropout
    input_dropout = 0.7

    # RNN output dropout
    rnn_output_dropout = 0.8

    # RNN state dropout
    rnn_state_dropout = 0.8

