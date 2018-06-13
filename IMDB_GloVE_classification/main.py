#!/usr/bin/python
# encoding: utf-8
"""
main
Purpose: Binary Classification with IMDB dataset using GloVe Embedding
Data: IMDB dataset (imported with Keras)
Input: 50,000 reviews of movies, labeled with 0 [negativ rating] or 1 [positiv rating]
Model: Sequential (Embedding + Dense), developed with Keras
Output: Probability of positiv rating -> Classification 1 [positiv rating] if proba>0.5 else 0 [negativ rating]

Author: datadonk23 (datadonk23@gmail.com)
Date: 2018-06-01
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend

from preprocessing import *

def main():
    """
    Main Loop

    :return: -
    """
    # Setup
    backend.clear_session()


    # Preprocess data
    print("... Preprocessing data ...")
    texts, labels = preprocess_labels(data_dir_path="data/aclImdb", dataset="train")
    vectorized_texts, word_index = tokenize_data(texts)
    X_train, y_train, X_val, y_val = split_data(vectorized_texts, labels)

    print("Data loaded:")
    print("Training features of shape {}".format(X_train.shape))
    print("Training labels of shape {}".format(y_train.shape))
    print("Validation features of shape {}".format(X_val.shape))
    print("Validation labels of shape {}".format(y_val.shape))


    # Preprocess GloVe embedding
    print("... Preprocessing embedding ...")
    embedding_idx = parse_glove(glove_file_path="model/glove.6B/glove.6B.100d.txt")
    embedding_matrix = make_embedding_matrix(embedding_idx, word_index)

    print("Embedding matrix of shape {} loaded".format(embedding_matrix.shape))


    # Model
    #TODO


if __name__ == "__main__":
    main()