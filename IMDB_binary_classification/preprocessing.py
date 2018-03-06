"""
    Preprocessing Data
"""
import numpy as np
from keras.datasets import imdb


class Data:
    """
        Data: IMDB ratings
        Properties:
            train_dev_test - loaded & preprocessed train, dev & test set
        Methods:
            load() - loads data with Keras
            one_hot_encode(sequences, dimension) - helper, one-hot-encode sequences
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
        Loads train and test data. Constrain: Only the 10000 most frequent words in reviews are used
        Sets _dataset to (train_data, train_labels), (test_data, test_labels)

        :return: -
        """
        self._dataset = imdb.load_data(num_words=10000)


    def preprocess(self):
        """
        Data wrangling: One-hot-encode data, vectorize labels
        Sets _dataset to encoded vectors - (train_data, train_labels), (test_data, test_labels)

        :return: -
        """
        def one_hot_encode(sequences, dimension=10000):
            """ Helper, one-hot-encoder for sequences """
            results = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                results[i, sequence] = 1.
            return results

        encoded_train_data = one_hot_encode(self._dataset[0][0])
        encoded_test_data = one_hot_encode(self._dataset[1][0])

        vectorized_train_labels = np.asarray(self._dataset[0][1]).astype("float32")  # vectorize train_labels
        vectorized_test_labels = np.asarray(self._dataset[1][1]).astype("float32")  # vectorize test_labels

        self._dataset = (encoded_train_data, vectorized_train_labels), (encoded_test_data, vectorized_test_labels)


    def split_data(self):
        """
        Split data into train, dev & test set
        Sets _dataset to train, dev & test Tuples (data, labels)

        :return: -
        """
        (train_data, train_labels) = (self._dataset[0][0][10000:] , self._dataset[0][1][10000:]) # 15000 samples
        (dev_data, dev_labels) = (self._dataset[0][0][:10000] , self._dataset[0][1][:10000]) # 10000 samples
        (test_data, test_labels) = self._dataset[1] # 25000 samples

        self._dataset = (train_data, train_labels), (dev_data, dev_labels), (test_data, test_labels)
