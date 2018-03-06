"""
    Model: Sequential
"""
import os
import numpy as np

from keras import models, layers, losses, metrics

class Model:
    """
        Model: NN with FC Layers
        Methods:
            build - model architecture
            train - model training
            evaluate - model evaluation
                save - helper, saves model
                load - helper, loads model
            predict - predicts label based on given input
    """
    def __init__(self):
        self.model = self.build()
        self.train_history = None
        self.model_dir = os.path.join(os.path.dirname(__file__), "model/")


    def build(self):
        """
        Architecture of Model

        :return: Keras model
        """
        input = layers.Input(shape=(10000, ))
        l = layers.Dense (16, activation="relu")(input)
        l = layers.Dense(16, activation="relu")(l)
        output = layers.Dense(1, activation="sigmoid")(l)

        self.model = models.Model(inputs=input, outputs=output)
        self.model.compile(optimizer="rmsprop",
                           loss=losses.binary_crossentropy,
                           metrics=[metrics.binary_accuracy])
        #print("Model Summary:")
        #print(self.model.summary())

        return self.model


    def train(self, train_set, dev_set, epochs=1, batch_size=512, save_model=True):
        """
        Model training - fit model with train data and validate on dev data

        :param train_set: (train_data, train_labels)
        :param dev_set: (dev_data, dev_labels)
        :param epochs: number of training epochs
        :param batch_size: number of samples for one pass
        :return: (train_history, model) - Tuple of monitoring metrices and trained model
        """
        train_data, train_labels = train_set
        dev_data, dev_labels = dev_set

        self.train_history = self.model.fit(x=train_data, y=train_labels,
                                            batch_size=batch_size, epochs=epochs,
                                            validation_data=(dev_data, dev_labels))
        if save_model:
            self.save(self.model, self.model_dir)

        print("Model trained successfully")

        return self.train_history.history, self.model


    def evaluate(self, test_set):
        """
        Evaluate trained model on test set

        :param test_set: (test_data, test_labels)
        :return: (loss, accuracy)
        """
        test_data, test_labels = test_set

        loss, acc = self.model.evaluate(test_data, test_labels)

        return loss, acc


    def save(self, model, dir):
        """
        Helper, saves model to disk

        :param model: Keras model instance
        :param dir: directory name
        """
        f_name = model.name + ".h5"
        model.save(os.path.join(dir, f_name))


    def load(self, dir, model_name):
        """
        Helper, loads persisted model into self.model

        :param dir: directory path of model
        :param model_name: name of persited Keras model
        :return: Keras model
        """
        model_path = os.path.join(dir, model_name)
        self.model = models.load_model(model_path)

        return self.model


    def predict(self, input, load_model_name=None):
        """
        Outputs a prediction

        :param input: input data, must be a np.array of shape (n, 10000) with values 0 or 1
        :param load_model_name: string with name of model (e.g. "model_1.h5"),
                                    default None (use self.model for prediction)
        :return: output = array of labels, either 0 [negativ rating] or 1 [positiv rating]
        """
        if load_model_name:
            self.model = self.load(self.model_dir, load_model_name)

        output_proba = self.model.predict(input)
        output_labels = np.where(output_proba > 0.5, 1, 0)

        return output_labels
