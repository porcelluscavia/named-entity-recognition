# Author 1: Jay Kejriwal, Martrikelnummer:4142919
# Author 2: Samantha Tureski, Martrikelnummer:4109680

#Honor Code:  We pledge that this program represents our own work.



from enum import Enum

class DefaultConfig:
    n_epochs = 10
    start_lr = 0.01
    # Decay Steps
    decay_step = 177
    # Decay Rate
    decay_rate = 0.9
    batch_size = 256
    max_timesteps = 30

    #Word embedding size
    word_embed_size = 40

    # Output sizes of the RNN layers.
    hidden_sizes = 350

    # Character embedding dropout
    input_dropout = 0.9

    # RNN output dropout
    rnn_output_dropout = 0.9

    # RNN state dropout
    rnn_state_dropout = 0.9


