import unittest
import numpy as np

from preprocessing import load_data, pad_texts


class ModelUtilsTests(unittest.TestCase):

    def setUp(self):
        self.mock_texts = np.array([[0, 0, 0, 0, 3, 1, 2], [0, 0, 0, 0, 4, 1, 2],
                                    [0, 0, 0, 0, 3, 1, 2], [0, 0, 0, 0, 4, 1, 2]])


    def tearDown(self):
        pass


    def test_load_data(self):
        """
        Test splitting of ratings into texts and labels lists
        """
        (X_train, y_train), (X_test, y_test) = load_data()

        self.assertEqual((25000,), X_train.shape, "Incorrect shape of training features")
        self.assertEqual((25000,), X_test.shape, "Incorrect shape of test features")
        self.assertEqual((25000,), y_train.shape, "Incorrect shape of training labels")
        self.assertEqual((25000,), y_test.shape, "Incorrect shape of test labels")

        self.assertEqual(list, X_train.dtype, "Incorrect type of values in training features")
        self.assertEqual(list, X_test.dtype, "Incorrect type of values in test features")
        self.assertEqual(np.int64, y_train.dtype, "Incorrect type of values in training labels")
        self.assertEqual(np.int64, y_test.dtype, "Incorrect type of values in test labels")

        self.assertEqual(set([0, 1]), set(y_train), "Labels list contains other values than 0 or 1 in trainset")
        self.assertEqual(set([0, 1]), set(y_test), "Labels list contains other values than 0 or 1 in testset")


    def test_pad_texts(self):
        """
        Test paddings
        """
        padded_sequences = pad_texts(self.mock_texts, 10)

        self.assertEqual((4, 10), padded_sequences.shape, "Incorrect shape of padded texts")
        self.assertEqual(np.int32, padded_sequences.dtype, "Incorrect type of values in padded sequences")
