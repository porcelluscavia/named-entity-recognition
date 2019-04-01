# Author 1: Jay Kejriwal, Martrikelnummer:4142919
# Author 2: Samantha Tureski, Martrikelnummer:4109680

#Honor Code:  We pledge that this program represents our own work.

from enum import Enum

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
            n_chars,
            embedding_size=64,
            phase=Phase.Predict):
        batch_size = batch.shape[1]
        input_size = batch.shape[2]
        label_size = label_batch.shape[2]

        self._x = tf.placeholder(tf.int64, shape=[batch_size, input_size])
        self._lens = tf.placeholder(tf.int64, shape=[batch_size])

        if phase != Phase.Predict:
            self._y = tf.placeholder(
                tf.int64, shape=[batch_size, label_size])

        embeddings = tf.get_variable("embeddings", shape=[n_chars, embedding_size])
        input_layer = tf.nn.embedding_lookup(embeddings, self._x)
        input_layer = tf.nn.dropout(input_layer, 0.95)
        
        f_cell = rnn.GRUCell(config.hidden_sizes)
        if phase == Phase.Train:
            f_cell = rnn.DropoutWrapper(f_cell,output_keep_prob=config.rnn_output_dropout, state_keep_prob=config.rnn_state_dropout)
        b_cell = rnn.GRUCell(config.hidden_sizes)
        if phase == Phase.Train:
            b_cell = rnn.DropoutWrapper(b_cell,output_keep_prob=config.rnn_output_dropout, state_keep_prob=config.rnn_state_dropout)
        (output_fw, output_bw),_ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, input_layer, sequence_length=self._lens, dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        weights = tf.get_variable("W", shape=[2*config.hidden_sizes, label_size])
        bias = tf.get_variable("b", shape=[label_size])
        timesteps = tf.shape(output)[1]
        output_flat = tf.reshape(output,[-1,2*config.hidden_sizes])
        logit = tf.matmul(output_flat, weights) + bias
        logits = tf.reshape(logit,[-1,timesteps,label_size])
        print(logits.shape)
        
        if phase == Phase.Train or Phase.Validation:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._y, logits=logits)
            mask =  tf.sequence_mask(self._lens)
            losses = tf.boolean_mask(losses, mask)
            self._loss = loss = tf.reduce_sum(losses)

        if phase == Phase.Train:
            step = tf.Variable(0, trainable=False)
            start_lr = tf.train.exponential_decay(0.1, step, 1, 0.5)

            self._train_op = tf.train.AdamOptimizer(start_lr) \
                .minimize(losses)
            self._probs = probs = tf.nn.softmax(logits)

        if phase == Phase.Validation:
#            hp_labels = tf.argmax(self._y, axis=1)

#            self._labels = tf.argmax(logits, axis=2)
            correct_prediction = tf.equal(tf.argmax(logits,1), self._y)

#            correct = tf.equal(hp_labels, self._labels)
            correct = tf.cast(correct_prediction, tf.float64)
            self._accuracy = tf.reduce_mean(correct)

    @property
    def accuracy(self):
        return self._accuracy


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
    def train_op(self):
        return self._train_op

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
