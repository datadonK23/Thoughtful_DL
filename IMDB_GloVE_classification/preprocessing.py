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
    :return: np.ndarray of shape (len(sequences), max_len)
    """
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    return padded_sequences


def split_data():
    """
    Split data into train and validation splits
    :return:
    """
    pass
