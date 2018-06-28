#!/usr/bin/python
# encoding: utf-8
"""
preprocessing
Purpose: Preprocessing IMDb dataset

Author: datadonk23 (datadonk23@gmail.com)
Date: 2018-06-27
"""
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences


def load_data(max_features=2000):
    """
    Load IMDB dataset with Keras
    :param max_features: number of words (considered as features)
    :return: trainset (X_train, y_train), testset (X_test, y_test)
    """
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

    return (X_train, y_train), (X_test, y_test)


def pad_texts(texts, max_len=500):
    """
    Pad sequences to given length
    :param texts: tokenized texts - ndarray of shape (N,)
    :param max_len: max length of sequence (either cut-off or fill in with 0)
    :return: ndarray of shape (N, max_len)
    """
    return pad_sequences(texts, maxlen=max_len)
