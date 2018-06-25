import unittest
import keras

from main import *

class MainTests(unittest.TestCase):

    def setUp(self):
        # keras backend
        self.assertEqual("tensorflow", keras.backend.backend(), "Keras using Theano")


if __name__ == "__main__":
    unittest.main()