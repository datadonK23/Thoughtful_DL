import unittest
import keras

from main import *

class MainTests(unittest.TestCase):

    def test_backend(self):
        # keras backend
        self.assertEqual("tensorflow", keras.backend.backend(), "Keras using Theano")


if __name__ == "__main__":
    unittest.main()
