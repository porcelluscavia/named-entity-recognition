# Author 1: Jay Kejriwal, Martrikelnummer:4142919
# Author 2: Samantha Tureski, Martrikelnummer:4109680

#Honor Code:  We pledge that this program represents our own work.

from enum import Enum
from sklearn.metrics import precision_recall_fscore_support as score

import tensorflow as tf
from tensorflow.contrib import rnn

class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2

class Model:
    def __init__(
            self,
            config,
            batch,
            lens_batch,
            label_batch,
            n_words,
            embedding_size=64,
            phase=Phase.Predict):
        batch_size = batch.shape[1]
        input_size = batch.shape[2]
        label_size = label_batch.shape[2]

        # The integer-encoded words. input_size is the (maximum) number of
        # time steps.
        self._x = tf.placeholder(tf.int64, shape=[batch_size, input_size])

        # This tensor provides the actual number of time steps for each
        # instance.
        self._lens = tf.placeholder(tf.int64, shape=[batch_size])

        # The integer-encoded tags. input_size is the (maximum) number of
        # time steps.
        if phase != Phase.Predict:
            self._y = tf.placeholder(
                tf.int64, shape=[batch_size, label_size])

        # Create an embedding matrix and look up each word.
        embeddings = tf.get_variable("embeddings", shape=[n_words, embedding_size])
        input_layer = tf.nn.embedding_lookup(embeddings, self._x)
        if phase == Phase.Train:
            input_layer = tf.nn.dropout(input_layer, config.input_dropout)

        # Apply one or more bidirectional GRU layers to the inputs.
        f_cell = rnn.GRUCell(config.hidden_sizes)
        if phase == Phase.Train:
            f_cell = rnn.DropoutWrapper(f_cell,output_keep_prob=config.rnn_output_dropout, state_keep_prob=config.rnn_state_dropout)
        b_cell = rnn.GRUCell(config.hidden_sizes)
        if phase == Phase.Train:
            b_cell = rnn.DropoutWrapper(b_cell,output_keep_prob=config.rnn_output_dropout, state_keep_prob=config.rnn_state_dropout)
        
        (output_fw, output_bw),_ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, input_layer, sequence_length=self._lens, dtype=tf.float32)

        # Concatenate the forward and backward representations.
        output = tf.concat([output_fw, output_bw], axis=-1)


        # Define weights and bias.
        weights = tf.get_variable("W", shape=[2*config.hidden_sizes, label_size])
        bias = tf.get_variable("b", shape=[label_size])

        #Storing the shape of valid time steps
        timesteps = tf.shape(output)[1]

        #Flatten the output 
        output_flat = tf.reshape(output,[-1,2*config.hidden_sizes])

        #Calculating Logits
        logit = tf.matmul(output_flat, weights) + bias

        #Reshaping back to original shape
        logits = tf.reshape(logit,[-1,timesteps,label_size])
        
        if phase == Phase.Train or Phase.Validation:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._y, logits=logits)
            #mask =  tf.sequence_mask(self._lens)
            #losses = tf.boolean_mask(losses, mask)
            self._loss = loss = tf.reduce_sum(losses)

        if phase == Phase.Train:
            step = tf.Variable(0, trainable=False)
            start_lr = tf.train.exponential_decay(config.start_lr, global_step=step,decay_steps=config.decay_step, decay_rate=config.decay_rate)

            self._train_op = tf.train.AdamOptimizer(start_lr) \
                .minimize(losses, global_step=step)
            self._probs = probs = tf.nn.softmax(logits)

        if phase == Phase.Validation:
            # Predicted labels
            self._plabels = tf.argmax(logits, axis=-1)
            
            # Correct labels
            correct_prediction = tf.equal(self.plabels, self._y)
            

            correct = tf.cast(correct_prediction, tf.float64)
            self._accuracy = tf.reduce_mean(correct)



            # Calculating precision 
            _, pop = tf.metrics.precision(labels=self._y,predictions=self._plabels)

            # Calculating recall
            _, rop = tf.metrics.recall(labels=self._y,predictions=self._plabels)

            # Calculating F1 score
            self._precision = pop
            self._recall = rop

            self._f1 = 2 * self._precision * self._recall / (self._precision + self._recall)
            

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def precision(self):
        return self._precision

    @property
    def recall(self):
        return self._recall

    @property
    def f1(self):
        return self._f1

    @property
    def lens(self):
        return self._lens

    @property
    def loss(self):
        return self._loss

    @property
    def probs(self):
        return self._probs

    @property
    def plabels(self):
        return self._plabels

    @property
    def train_op(self):
        return self._train_op

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
