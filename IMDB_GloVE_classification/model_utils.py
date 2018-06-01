#!/usr/bin/python
# encoding: utf-8
"""
model
Purpose: Provide functions to build and persist model

Author: datadonk23 (datadonk23@gmail.com)
Date: 2018-06-01
"""
import os
from keras import models, layers


def build_model():
    """
    Builds sequential model

    :return: model
    """
    model = models.Sequential()
    model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])

    return model


def save_model(model, dir):
    """
    Helper, saves model to disk

    :param model: Keras model instance
    :param dir: directory name
    """
    f_name = model.name + ".h5"
    model.save(os.path.join(dir, f_name))


def load_model(dir, model_name):
    """
    Helper, loads persisted model

    :param dir: directory path of model
    :param model_name: name of persited Keras model
    :return: model
    """
    model_path = os.path.join(dir, model_name)
    model = models.load_model(model_path)

    return model
