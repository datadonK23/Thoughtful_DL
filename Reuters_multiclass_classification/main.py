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

    model = Model()

    history, trained_model = model.train((train_data, train_labels), (dev_data, dev_labels), epochs=20)
    print(type(history))
    print(history["val_categorical_accuracy"])
    # FIXME asset < threshold

    loss, acc = model.evaluate((test_data, test_labels))
    print("Loss: " + str(loss))
    print("Accuracy: " + str(acc))



if __name__ == "__main__":
    main()