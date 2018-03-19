"""
    Multiclass Classification with newswire dataset
    Data: Reuters dataset (imported with Keras), orig 1986
    Input: 11,228 newswires, annotated with 46 different topics
    Model: Sequential, developed with Keras
    Output: Probability of #FIXME
"""
from keras import backend

from model import *
from preprocessing import *


def main():
    """
    Main Loop

    :return: -
    """
    backend.clear_session()

    (train_data, train_labels), (dev_data, dev_labels),  (test_data, test_labels) = Data().train_dev_test

    # Model exploration
    model_explore = Model()
    model_explore.train((train_data, train_labels), (dev_data, dev_labels), epochs=20)
    model_explore.plot_history()

    # Training final model
    model_final = Model()
    model_final.train((train_data, train_labels), (dev_data, dev_labels), epochs=9)
    loss, acc = model_final.evaluate((test_data, test_labels))
    print("Loss: " + str(loss))
    print("Accuracy: " + str(acc))


if __name__ == "__main__":
    main()