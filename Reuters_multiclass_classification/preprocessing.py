"""
    Preprocessing Data
"""
import numpy as np
from keras.datasets import reuters
from keras.utils import to_categorical

class Data:
    """
        Data: Reuters newswires
        Properties:
            train_dev_test - train, dev & test set
        Methods:
            load() - loads data with Keras
            preprocess() - data wrangling
            split_data() - splits data into train, dev & test sets
    """
    def __init__(self):
        self._dataset = None


    @property
    def train_dev_test(self):
        self.load()
        self.preprocess()
        self.split_data()

        return self._dataset


    def load(self):
        """
        Loads train and test data. Constrain: Only the 10000 most frequent words in newswires are used
        Sets _dataset to (train_data, train_labels), (test_data, test_labels)

        :return: -
        """
        self._dataset = reuters.load_data(num_words=10000)


    def preprocess(self):
        """
        Data wrangling: One-hot-encode data & labels
        Sets _dataset to encoded vectors - (train_data, train_labels), (test_data, test_labels)

        :return: -
        """
        def one_hot_encode(sequences, dimension=10000):
            results = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                results[i, sequence] = 1.
            return results

        encoded_train_data = one_hot_encode(self._dataset[0][0])
        encoded_test_data = one_hot_encode(self._dataset[1][0])

        encoded_train_labels = to_categorical(self._dataset[0][1])
        encoded_test_labels = to_categorical(self._dataset[1][1])

        self._dataset = (encoded_train_data, encoded_train_labels), (encoded_test_data, encoded_test_labels)


    def split_data(self):
        """
        Split data into train, dev & test set
        Sets _dataset to train, dev & test Tuples (data, labels)

        :return: -
        """
        (train_data, train_labels) = (self._dataset[0][0][1000:] , self._dataset[0][1][1000:]) # 7982 samples
        (dev_data, dev_labels) = (self._dataset[0][0][:1000] , self._dataset[0][1][:1000]) # 1000 samples
        (test_data, test_labels) = self._dataset[1] # 2246 samples

        self._dataset = (train_data, train_labels), (dev_data, dev_labels), (test_data, test_labels)
