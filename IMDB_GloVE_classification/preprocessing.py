#!/usr/bin/python
# encoding: utf-8
"""
preprocessing
Purpose: Preprocessing IMDb dataset

Author: datadonk23 (datadonk23@gmail.com)
Date: 2018-06-12
"""
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def preprocess_labels(data_dir_path="data/aclImdb", dataset="train"):
    """
    Parses aclImdb data directory and generates two lists. One contains text of the review and other the corresponding
    label, which is 0 for negative and 1 for positive ratings.
    :param data_dir_path: String, path to dataset directory
    :param dataset: String, either "train" for training or "test" for testing dataset
    :return: texts=[review_text], labels=np.ndarray([0|1]) - label: 0 = "neg", 1 = "pos" rating
    """
    if dataset not in ["train", "test"]:
        raise ValueError('Specify correct dataset: either "train" or "test"')

    dataset_dir = os.path.join(data_dir_path, dataset)
    texts, labels = [], []

    for label_type in ["neg", "pos"]:
        dir_name = os.path.join(dataset_dir, label_type)
        for file_name in os.listdir(dir_name):
            if file_name[-4:] == ".txt": # process txt files only
                file = os.path.join(dir_name, file_name)
                with open(file) as f:
                    texts.append(f.read())
                if label_type == "neg":
                    labels.append(0)
                else:
                    labels.append(1)
    labels = np.asarray(labels)

    return texts, labels


def tokenize_data(texts, max_len=100, max_words=1000):
    """
    Vectorize reviews
    :param texts: list of review texts
    :param max_len: max length of review text
    :param max_words: max number of words to consider in dataset
    :return: tokenized_sequences=np.ndarray of shape (len(sequences), max_len), word_index=Dict{word: encoding}
    """
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    return padded_sequences, word_index


def split_data(vectorized_texts, labels, train_samples=200, val_samples=100000):
    """
    Split data into train and validation splits
    :param vectorized_texts: np.ndarray of shape (len(sequences), max_len)
    :param labels: np.ndarray([0|1])
    :param train_samples: number of training samples
    :param val_samples: number of validation samples
    :return: X_train, y_train, X_val, y_val - np.ndarray each
    """
    indices = np.arange(vectorized_texts.shape[0])
    np.random.shuffle(indices)

    data = vectorized_texts[indices]
    labels = labels[indices]

    X_train = data[:train_samples]
    y_train = labels[:train_samples]
    X_val = data[train_samples : train_samples + val_samples]
    y_val = labels[train_samples: train_samples + val_samples]

    return X_train, y_train, X_val, y_val


def parse_glove(glove_file_path="model/glove.6B/glove.6B.100d.txt"):
    """
    Parse GloVe file and create embedding index
    :param glove_file_path: path to pretrained GloVe embedding
    :return: embedding_idx - Dict{word: [coefs]}
    """
    embedding_idx = {}
    with open(glove_file_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1: ], dtype="float32")
            embedding_idx[word] = coefs

    return embedding_idx


def make_embedding_matrix(embedding_idx, word_index, max_words=1000, embedding_dim=100):
    """
    Generate GloVe word-embedding matrix
    :param embedding_idx: Dict{word: [coefs]} parsed from GloVe file
    :param word_index: word_index=Dict{word: encoding}
    :param max_words: max number of words to consider in dataset
    :param embedding_dim: dimension of GloVe embedding (50, 100, 200, 300 for models trained on Wikipedia dataset)
    :return: embedding_matrix - np.ndarray
    """
    shape = (max_words, embedding_dim)
    embedding_matrix = np.zeros(shape)
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embedding_idx.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix
