#!/usr/bin/python
# encoding: utf-8
"""
main
Purpose: Binary Classification with IMDB dataset using GloVe Embedding
Data: IMDB dataset (imported with Keras)
Input: 50,000 reviews of movies, labeled with 0 [negativ rating] or 1 [positiv rating]
Model: Sequential (Embedding + Dense), developed with Keras

Author: datadonk23 (datadonk23@gmail.com)
Date: 2018-06-01
"""
from keras import backend

from preprocessing import *
from model import *


def main():
    """
    Main logic

    :return: -
    """
    # Setup
    backend.clear_session()


    # Preprocess data
    print("... Preprocessing data ...")
    texts, labels = preprocess_labels(data_dir_path="data/aclImdb", dataset="train")
    test_texts, y_test = preprocess_labels(data_dir_path="data/aclImdb", dataset="test")
    vectorized_texts, word_index = tokenize_data(texts)
    X_test, _ = tokenize_data(test_texts)
    X_train, y_train, X_val, y_val = split_data(vectorized_texts, labels, train_samples=2500)

    print("Data loaded:")
    print("Training features of shape {}".format(X_train.shape))
    print("Training labels of shape {}".format(y_train.shape))
    print("Validation features of shape {}".format(X_val.shape))
    print("Validation labels of shape {}".format(y_val.shape))
    print("Test features of shape {}".format(X_test.shape))
    print("Test labels of shape {}".format(y_test.shape))


    # Preprocess GloVe embedding
    print("... Preprocessing embedding ...")
    embedding_idx = parse_glove(glove_file_path="models/glove.6B/glove.6B.100d.txt")
    embedding_matrix = make_embedding_matrix(embedding_idx, word_index)

    print("Embedding matrix of shape {} loaded".format(embedding_matrix.shape))


    # Model
    print("... Building model ...")
    model = build_model(embedding_matrix)
    plot_model(model)

    print("... Training model ...")
    history, trained_model= train_model(model, (X_train, y_train), (X_val, y_val), epochs=9)
    save_model(trained_model, "models/")
    plot_history(history)


    # Evaluation
    print("... Evaluating model ...")
    loss, acc = evaluate_model(trained_model, (X_test, y_test))

    print("Test scores:")
    print("Loss", loss)
    print("Accuracy", acc)


if __name__ == "__main__":
    main()