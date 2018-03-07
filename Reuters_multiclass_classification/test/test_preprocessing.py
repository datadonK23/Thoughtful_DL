import unittest
import numpy as np

from preprocessing import Data


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.data = Data()


    def test_load(self):
        """
        Test existence, type, length of loaded data
        """
        self.data.load()

        self.assertIsNotNone(self.data._dataset, "loaded no data")
        self.assertEqual(type(("foo", "bar")), type(self.data._dataset), "loaded no tuple")
        self.assertEqual(2, len(self.data._dataset), "loaded tuple has false length")


    def test_preprocess(self):
        """
        Test one-hot-encding and type conversions of preprocessed data
        """
        self.data.load()
        self.data.preprocess()

        # test features one-hot-encoder
        np.testing.assert_array_equal([0., 1.], np.unique(self.data._dataset[0][0]),
                                      "false one-hot-encoding of train_data")
        np.testing.assert_array_equal([0., 1.], np.unique(self.data._dataset[1][0]),
                                      "false one-hot-encoding of test_data")
        self.assertEqual("float64", self.data._dataset[0][0].dtype, "wrong type of train_data values")
        self.assertEqual("float64", self.data._dataset[1][0].dtype, "wrong type of test_data values")

        # test label vectorization
        np.testing.assert_array_equal([0., 1.], np.unique(self.data._dataset[0][1]),
                                      "false one-hot-encoding of train_labels")
        np.testing.assert_array_equal([0., 1.], np.unique(self.data._dataset[1][1]),
                                      "false one-hot-encoding of test_labels")
        self.assertEqual("float64", self.data._dataset[0][1].dtype, "wrong type of train_labels values")
        self.assertEqual("float64", self.data._dataset[1][1].dtype, "wrong type of test_labels values")


    def test_split_data(self):
        """
        Test correct train-dev-test-split
        """
        self.data.load()
        self.data.preprocess()
        self.data.split_data()

        # correct number of tuples
        self.assertEqual(3, len(self.data._dataset), "wrong number of splits")
        self.assertEqual(2, len(self.data._dataset[0]), "wrong number of train splits")
        self.assertEqual(2, len(self.data._dataset[1]), "wrong number of dev splits")
        self.assertEqual(2, len(self.data._dataset[2]), "wrong number of test splits")

        # existence
        self.assertIsNotNone(self.data._dataset[0][0], "train_data is None")
        self.assertIsNotNone(self.data._dataset[0][1], "train_labels is None")
        self.assertIsNotNone(self.data._dataset[1][0], "dev_data is None")
        self.assertIsNotNone(self.data._dataset[1][1], "dev_labels is None")
        self.assertIsNotNone(self.data._dataset[2][0], "test_data is None")
        self.assertIsNotNone(self.data._dataset[2][1], "test_labels is None")


    def test_train_dev_test(self):
        """
        Test type and shape of train, dev & test sets
        """
        (train_data, train_labels), (dev_data, dev_labels), (test_data, test_labels) = self.data.train_dev_test

        # type
        self.assertEqual(np.ndarray, type(train_data), "wrong type of train_data")
        self.assertEqual(np.ndarray, type(train_labels), "wrong type of train_labels")
        self.assertEqual(np.ndarray, type(dev_data), "wrong type of dev_data")
        self.assertEqual(np.ndarray, type(dev_labels), "wrong type of dev_labels")
        self.assertEqual(np.ndarray, type(test_data), "wrong type of test_data")
        self.assertEqual(np.ndarray, type(test_labels), "wrong type of test_labels")

        # shape
        self.assertEqual((7982, 10000), train_data.shape, "train_data has wrong shape")
        self.assertEqual((7982, 46), train_labels.shape, "train_labels have wrong shape")
        self.assertEqual((1000, 10000), dev_data.shape, "dev_data has wrong shape")
        self.assertEqual((1000, 46), dev_labels.shape, "dev_labels have wrong shape")
        self.assertEqual((2246, 10000), test_data.shape, "test_data has wrong shape")
        self.assertEqual((2246, 46), test_labels.shape, "test_labels have wrong shape")


if __name__ == "__main__":
    unittest.main()
