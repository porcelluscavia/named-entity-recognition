from enum import Enum
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn import metrics

from config import DefaultConfig
from model import Model, Phase
from numberer import Numberer


def read_token_and_tags(filename):
    words = []
    tags = []
    temp_words = [] # this stores the words for the current sentence only
    temp_tags = []
    #add command line here
    input = open(filename, encoding='utf8')
    for line in input.read().split("\n") :
        # if line is blank, add temp lists and reset
        if line.strip() == "" :
            words += [temp_words]
            tags += [temp_tags]
            temp_words = []
            temp_tags = []
        # else split by whitespace and add to lists
        else :
            fields = line.split()
            temp_words += [fields[1]]
            temp_tags += [fields[5]]
    #print(words)
    #print(tags)
    return(words, tags)

def recode_token_and_tags(token_and_tags, words, tags, train=False):
    int_word_token = []
    int_tag_token = []
    int_all = []

    for x in range(len(token_and_tags[0])):
        int_word = []
        for word in token_and_tags[0][x]:
            int_word.append(words.number(word, train))
        int_word_token.append(int_word)
        
        int_tag = []
        for tag in token_and_tags[1][x]:
            int_tag.append(tags.number(tag, train))
        int_tag_token.append(int_tag)
            

    int_all.append(int_word_token)
    int_all.append(int_tag_token)
    #print(int_all)
    #return(int_word_token, int_tag_token)
    return (int_all)
    
def generate_instances(
    data,
    max_timesteps,
    batch_size=128):    
    
    n_batches=(len(data[0]) // batch_size)
    
    # We are discarding the last batch for now, for simplicity.
    
    labels = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_timesteps),
        dtype=np.int32)
    lengths = np.zeros(
        shape=(
            n_batches,
            batch_size),
        dtype=np.int32)
    words = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_timesteps),
        dtype=np.int32)
    for batch in range(n_batches):
        for idx in range(batch_size):
            
            word = data[0][(batch * batch_size) + idx]
            label = data[1][(batch * batch_size) + idx]
            
            # Add labels
            timesteps = min(max_timesteps, len(label),len(word))

            # Label with timesteps
            labels[batch, idx, :timesteps] = label[:timesteps]

            # Sequence length (time steps)
            lengths[batch, idx] = timesteps

            # Words with timesteps
            words[batch, idx, :timesteps] = word[:timesteps]
            

    return (words, lengths, labels)


def train_model(config, train_batches, validation_batches):
    train_batches, train_lens, train_labels = train_batches
    validation_batches, validation_lens, validation_labels = validation_batches

    n_words = max(np.amax(validation_batches), np.amax(train_batches)) + 1
    print(n_words)

    with tf.Session() as sess:
        with tf.variable_scope("model", reuse=False):
            train_model = Model(
                config,
                train_batches,
                train_lens,
                train_labels,
                n_words,
                phase=Phase.Train)

        with tf.variable_scope("model", reuse=True):
            validation_model = Model(
                config,
                validation_batches,
                validation_lens,
                validation_labels,
                n_words,
                phase=Phase.Validation)

        sess.run(tf.global_variables_initializer())

        for epoch in range(config.n_epochs):
            train_loss = 0.0
            validation_loss = 0.0
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0

            # Train on all batches.
            for batch in range(train_batches.shape[0]):
                loss, _ = sess.run([train_model.loss, train_model.train_op], {
                    train_model.x: train_batches[batch], train_model.lens: train_lens[batch], train_model.y: train_labels[batch]})
                train_loss += loss

            # validation on all batches.
            for batch in range(validation_batches.shape[0]):
                loss, acc, hpl = sess.run([validation_model.loss, validation_model.accuracy, validation_model.labels], {
                    validation_model.x: validation_batches[batch], validation_model.lens: validation_lens[batch], validation_model.y: validation_labels[batch]})
                validation_loss += loss
                accuracy += acc
                precision += metrics.precision_score(np.argmax(np.array(validation_labels[batch]).astype(np.int32), axis = 1), hpl, average = "macro")
                recall += metrics.recall_score(np.argmax(np.array(validation_labels[batch]).astype(np.int32), axis = 1), hpl, average = "macro")
                f1 += metrics.f1_score(np.argmax(np.array(validation_labels[batch]).astype(np.int32), axis = 1), hpl, average = "macro")

            train_loss /= train_batches.shape[0]
            validation_loss /= validation_batches.shape[0]
            accuracy /= validation_batches.shape[0]
            precision /= validation_batches.shape[0]
            recall /= validation_batches.shape[0]
            f1 /= validation_batches.shape[0]
            print(
                "epoch %d - train loss: %.2f, validation loss: %.2f, validation acc: %.2f precision: %.2f recall: %.2f F1: %.2f" %
                (epoch, train_loss, validation_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: %s TRAIN_SET DEV_SET\n" % sys.argv[0])
        sys.exit(1)

    config = DefaultConfig()

    # Read training and validation data.
    #train_sentence_tags = read_token_and_tags("part0.conll")
    #valid_sentence_tags = read_token_and_tags("part1.conll")

    train_sentence_tags = read_token_and_tags(sys.argv[1])
    valid_sentence_tags = read_token_and_tags(sys.argv[2])

    # Convert word characters and part-of-speech labels to numeral
    # representation.
    words = Numberer()
    tags = Numberer()
    train_sentence_tags = recode_token_and_tags(train_sentence_tags, words, tags, train=True)
    valid_sentence_tags = recode_token_and_tags(valid_sentence_tags, words, tags)

    # Generate batches
    train_batches = generate_instances(
        train_sentence_tags,
        config.max_timesteps,
        batch_size=config.batch_size)
    validation_batches = generate_instances(
        valid_sentence_tags,
        config.max_timesteps,
        batch_size=config.batch_size)

    #print(train_batches)

    # Train the model
    train_model(config, train_batches, validation_batches)
