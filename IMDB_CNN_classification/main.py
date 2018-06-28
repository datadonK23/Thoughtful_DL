#!/usr/bin/python
# encoding: utf-8
"""
main
Purpose: Sentiment analysis (binary classification) with IMDB dataset using a 1D CNN
Data: IMDB dataset (imported with Keras)
Input: 50,000 reviews of movies, labeled with 0 [negativ rating] or 1 [positiv rating]
Model: Sequential (Embedding + 2 * 1D CNN (max pool)  + Dense), developed with Keras

Author: datadonk23 (datadonk23@gmail.com)
Date: 2018-06-27
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


    # print("Data loaded:")
    # print("Training features of shape {}".format(X_train.shape))
    # print("Training labels of shape {}".format(y_train.shape))
    # print("Validation features of shape {}".format(X_val.shape))
    # print("Validation labels of shape {}".format(y_val.shape))
    # print("Test features of shape {}".format(X_test.shape))
    # print("Test labels of shape {}".format(y_test.shape))


    # Model
    print("... Building model ...")
    # model = build_model(embedding_matrix)
    # plot_model(model)

    print("... Training model ...")
    # history, trained_model= train_model(model, (X_train, y_train), (X_val, y_val), epochs=9)
    # save_model(trained_model, "models/")
    # plot_history(history)


    # Evaluation
    print("... Evaluating model ...")
    # loss, acc = evaluate_model(trained_model, (X_test, y_test))
    #
    # print("Test scores:")
    # print("Loss", loss)
    # print("Accuracy", acc)


if __name__ == "__main__":
    main()

