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
    print("... Load & preprocess data ...")
    (X_train, y_train), (X_test, y_test) = load_data()

    X_train = pad_texts(X_train)
    X_test = pad_texts(X_test)

    print("Data loaded & preprocessed:")
    print("Training features of shape {}".format(X_train.shape))
    print("Training labels of shape {}".format(y_train.shape))
    print("Test features of shape {}".format(X_test.shape))
    print("Test labels of shape {}".format(y_test.shape))


    # Model
    print("... Building model ...")
    model = build_model()
    plot_model(model)

    print("... Training model ...")
    print("Launch TensorBoard server in Terminal to monitor training: $ tensorboard --logdir='models/logs'")
    history, trained_model= train_model(model, (X_train, y_train))
    plot_history(history)


    # Evaluation
    print("... Evaluating model ...")
    loss, acc = evaluate_model(trained_model, (X_test, y_test))

    print("Test scores:")
    print("Loss", loss)
    print("Accuracy", acc)


if __name__ == "__main__":
    main()

