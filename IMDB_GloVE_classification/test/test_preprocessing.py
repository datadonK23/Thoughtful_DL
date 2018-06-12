import unittest
import os

from preprocessing import preprocess_labels


class ModelUtilsTests(unittest.TestCase):

    def setUp(self):
        self.mock_data_path = "data/mock_aclImdb"


    def tearDown(self):
        pass


    def test_process_labels(self):
        """
        Test splitting of ratings into texts and labels lists
        """
        texts_train, labels_train = preprocess_labels(self.mock_data_path, "train")
        texts_test, labels_test = preprocess_labels(self.mock_data_path, "train")

        self.assertTrue(texts_train, "Train texts not generated")
        self.assertTrue(labels_train, "Train labels not generated")
        self.assertTrue(texts_test, "Test texts not generated")
        self.assertTrue(labels_test, "Test labels not generated")

        self.assertEqual(set(labels_train), set([0, 1]), "Labels list contains not only 0 or 1 in trainset")
        self.assertEqual(set(labels_test), set([0, 1]), "Labels list contains not only 0 or 1 in testset")

        self.assertEqual(len(texts_train), len(labels_train), "Train texts and labels lists have different length")
        self.assertEqual(len(texts_test), len(labels_test), "Test texts and labels lists have different length")

