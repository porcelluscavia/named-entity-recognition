from enum import Enum

class DefaultConfig:
    n_epochs = 50
    start_lr = 0.0009
    decay_steps = 177
    decay_rate = 0.90

    batch_size = 32
    max_timesteps = 20

    word_embed_size = 12
    forward_cell_units = 25
    backward_cell_units = 25

    # Output sizes of the RNN layers.
    hidden_sizes = [250]

    hidden_dropout= 0.6
    # Character embedding dropout
    input_dropout = 0.7

    # RNN output dropout
    rnn_output_dropout = 0.8

    # RNN state dropout
    rnn_state_dropout = 0.9

