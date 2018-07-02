#!/usr/bin/python
# encoding: utf-8
"""
model
Purpose: Provide functions to build and persist models

Author: datadonk23 (datadonk23@gmail.com)
Date: 2018-06-27
"""
import os
from keras import models, layers, callbacks, utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def build_model():
    """
    Builds sequential model:
    Embedding -> CNN_1D [MaxPool] -> CNN_1D [MaxPool] -> Dense(1)
    :return: model
    """
    max_features = 2000
    max_len = 500

    model = models.Sequential()
    model.add(layers.Embedding(max_features, 128, input_length=max_len, name="embed"))
    model.add(layers.Conv1D(32, 7, activation="relu"))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation="relu"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])

    return model


def train_model(model, train_set, epochs=20, batch_size=128):
    """
    Model training - fit model with train data and validate on dev data
    :param model: keras model which should be trained
    :param train_set: (X_train, y_train)
    :param epochs: number of training epochs
    :param batch_size: number of samples for one pass
    :return: (train_history, model) - Tuple of monitoring metrices and trained model
    """
    X_train, y_train = train_set

    callbacks_list = [
        callbacks.EarlyStopping(monitor="acc", patience=1), # stops when accuracy doesn't improve for >1 (=2) epochs
        callbacks.ModelCheckpoint(filepath="models/best_model_training.h5",
                                  monitor="val_loss", save_best_only=True), # saves best trained model
        callbacks.TensorBoard(log_dir="models/logs", histogram_freq=1)
    ]

    train_history = model.fit(X_train, y_train,
                              epochs=epochs, batch_size=batch_size, validation_split=0.2,
                              callbacks=callbacks_list)
    num_epochs = len(train_history.history["loss"])

    print("Model trained successfully in {} epochs".format(num_epochs))

    return train_history.history, model


def evaluate_model(model, test_set):
    """
    Evaluates trained model on test set
    :param model: Trained keras model
    :param test_set: (X_test, y_test)
    :return: (loss, accuracy)
    """
    (X_test, y_test) = test_set
    loss, acc = model.evaluate(X_test, y_test)

    return loss, acc


def plot_model(model, dir="plots/"):
    """
    Helper, plots model with shape informations and saves it to disk
    :param model: keras model which should be plotted
    :param dir: directory to save the plot
    """
    f_name = model.name + ".png"
    f_path = os.path.join(dir, f_name)
    utils.plot_model(model, show_shapes=True, to_file=f_path)

    model_plot = mpimg.imread(f_path)
    plt.imshow(model_plot)
    plt.show()


def save_model(model, dir):
    """
    Helper, saves models to disk
    :param model: Keras models instance
    :param dir: directory name
    """
    f_name = model.name + ".h5"
    model.save(os.path.join(dir, f_name))


def load_model(dir, model_name):
    """
    Helper, loads persisted models
    :param dir: directory path of models
    :param model_name: name of persited Keras models
    :return: models
    """
    model_path = os.path.join(dir, model_name)
    model = models.load_model(model_path)

    return model


def plot_history(history):
    """
    Helper, plot loss and accuracy of trained model
    :param history - dict with history of trained model
    """
    loss = history["loss"]
    val_loss = history["val_loss"]
    acc = history["acc"]
    val_acc = history["val_acc"]

    epochs = range(1, len(loss) + 1)

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.legend()

    plt.subplots_adjust(hspace=0.5)
    plt.show()


