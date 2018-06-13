import unittest
import numpy as np

from preprocessing import preprocess_labels, tokenize_data, split_data, parse_glove, make_embedding_matrix


class ModelUtilsTests(unittest.TestCase):

    def setUp(self):
        self.mock_data_path = "data/mock_aclImdb"
        self.mock_texts = ["First mocked review.", "Second mocked review."]
        self.mock_token_vectors = np.array([[0, 0, 0, 0, 3, 1, 2], [0, 0, 0, 0, 4, 1, 2],
                                            [0, 0, 0, 0, 3, 1, 2], [0, 0, 0, 0, 4, 1, 2]])
        self.mock_labels = np.array([0, 1, 0, 1])


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
        Test tokenization of review data with var parameters
        """
        token100_vector, word_idx = tokenize_data(self.mock_texts)

        self.assertEqual(np.ndarray, type(token100_vector), "Tokenized data not a numpy array")
        self.assertEqual(2, len(token100_vector), "Incorrect number of returned sequences")
        self.assertEqual(dict, type(word_idx), "Incorrect type of word index")
        self.assertEqual(4, len(word_idx), "Incorrect length of word index dict")
        for k in word_idx.keys():
            self.assertEqual(str, type(k), "Incorrect type of word index dict keys")
        for v in word_idx.values():
            self.assertEqual(int, type(v), "Incorrect type of word index dict values")

        token20_vector, _ = tokenize_data(self.mock_texts, 20)
        self.assertEqual(20, len(token20_vector[0]), "Incorrect length of token vector")
        self.assertEqual(len(token20_vector[0]), len(token20_vector[1]), "Token vectors don't match on length")

        token20__wordlimit_vector, _ = tokenize_data(self.mock_texts, 20, 4)
        self.assertEqual(3, token20__wordlimit_vector.max(), "Incorrect number of words in word-limited vector")


    def test_split_data(self):
        """
        Test shape of train and validation sets
        """
        X_train, y_train, X_val, y_val = split_data(self.mock_token_vectors, self.mock_labels, 2, 2)

        self.assertEqual((2, 7), X_train.shape, "Incorrect shape of training samples after split")
        self.assertEqual((2, 7), X_val.shape, "Incorrect shape of validation samples after split")
        self.assertEqual((2,), y_train.shape, "Incorrect shape of training labels after split")
        self.assertEqual((2,), y_val.shape, "Incorrect shpae of validation labels after split")


    def test_parse_glove(self):
        """
        Test parsing of glove file
        """
        embedding_idx = parse_glove("model/mock_glove.6B/mock_glove.6B.50d.txt")

        self.assertEqual(type({}), type(embedding_idx), "Returned object is not a dict")
        self.assertEqual(7, len(embedding_idx), "Incorrect len of embedding_idx dict")

        for k in embedding_idx.keys():
            self.assertEqual(str, type(k), "Embedding index keys are not strings")
        for v in embedding_idx.values():
            self.assertEqual(np.float32, v.dtype, "Embedding index values are not float32")


    def test_make_embedding_matrix(self):
        """
        Test shape of embedding matrix
        """
        pass#matrix = make_embedding_matrix()