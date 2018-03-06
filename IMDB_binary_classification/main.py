"""
    Binary Classification with IMDB dataset
    Data: IMDB dataset (imported with Keras)
    Input: 50,000 reviews of movies, labeled with 0 [negativ rating] or 1 [positiv rating]
    Model: Sequential, developed with Keras
    Output: Probability of positiv rating -> Classification 1 [positiv rating] if proba>0.5 else 0 [negativ rating]
"""
from keras import backend

from model import *
from preprocessing import *


def main():
    """
    Main Loop

    :return: -
    """
    # Setup
    backend.clear_session()

    # Data
    (train_data, train_labels), (dev_data, dev_labels),  (test_data, test_labels) = Data().train_dev_test

    # Train
    model = Model()
    history, trained_model = model.train((train_data, train_labels), (dev_data, dev_labels), epochs=1)
    print("Loss: " + str(history["loss"]))
    print("Accuracy: " + str(history["binary_accuracy"]))
    print("Val loss: " + str(history["val_loss"]))
    print("Val accuracy: " + str(history["val_binary_accuracy"]))

    # Evaluate
    loss, acc = model.evaluate((test_data, test_labels))
    print("Loss: " + str(loss))
    print("Accuracy: " + str(acc))

    # Predict
    input = np.random.randint(2, size=20000).reshape(2, 10000)
    predictions = model.predict(input)
    print("Predictions:")
    for i, prediction in enumerate(predictions):
        print("Prediction for Rating {}: {}".format(i, "positiv" if prediction[0] == 1 else "negativ"))


if __name__ == "__main__":
    main()