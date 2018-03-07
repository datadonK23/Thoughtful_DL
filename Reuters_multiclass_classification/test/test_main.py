import unittest

import keras

import main


class MainTests(unittest.TestCase):
    def setUp(self):
        # keras backend
        self.assertEqual("tensorflow", keras.backend.backend(), "Keras using Theano")

    def test_main(self):
        unit = main.main()

        # main function should be side-effecting
        self.assertIsNone(unit, "main returns something")


if __name__ == "__main__":
    unittest.main()