import unittest
import numpy as np

from preprocessing import preprocess_labels, tokenize_data


class ModelUtilsTests(unittest.TestCase):

    def setUp(self):
        self.mock_data_path = "data/mock_aclImdb"
        self.mock_texts = ["First mocked review.", "Second mocked review."]


    def tearDown(self):
        pass


    def test_process_labels(self):
        """
        Test splitting of ratings into texts and labels lists
        """
        texts_train, labels_train = preprocess_labels(self.mock_data_path, "train")
        texts_test, labels_test = preprocess_labels(self.mock_data_path, "train")

        self.assertEqual(set([0, 1]), set(labels_train), "Labels list contains not only 0 or 1 in trainset")
        self.assertEqual(set([0, 1]), set(labels_test), "Labels list contains not only 0 or 1 in testset")

        self.assertEqual(len(labels_train), len(texts_train), "Train texts and labels lists have different length")
        self.assertEqual(len(labels_test), len(texts_test), "Test texts and labels lists have different length")

        self.assertEqual(np.ndarray, type(labels_train), "Train labels not a numpy array")


    def test_tokenize_data(self):
        """
        Test tokenization of review data
        """
        token100_vector = tokenize_data(self.mock_texts)

        self.assertEqual(np.ndarray, type(token100_vector), "Tokenized data not a numpy array")
        self.assertEqual(2, len(token100_vector), "Incorrect number of returned sequences")

        token20_vector = tokenize_data(self.mock_texts, 20)
        self.assertEqual(20, len(token20_vector[0]), "Incorrect length of token vector")
        self.assertEqual(len(token20_vector[0]), len(token20_vector[1]), "Token vectors don't match on length")

        token20__wordlimit_vector = tokenize_data(self.mock_texts, 20, 4)
        self.assertEqual(3, token20__wordlimit_vector.max(), "Incorrect number of words in word-limited vector")


    def test_split_data(self):
        """
        Test train and validation split
        """
        pass