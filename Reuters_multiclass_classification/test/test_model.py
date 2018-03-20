import unittest
import numpy as np
import json
import shutil, tempfile

from model import Model


class ModelTests(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        self.test_model_dir = tempfile.mkdtemp()
        self.mock_train_set = (np.zeros((1, 10000)), np.zeros((1, 46)))
        self.mock_dev_set = self.mock_train_set
        self.mock_test_set = self.mock_train_set
        self.mock_input = np.random.randint(2, size=20000).reshape(2, 10000)


    def tearDown(self):
        shutil.rmtree(self.test_model_dir)


    def test_build(self):
        """
        Test architecture of model
        """
        built_model = self.model.build()
        j_model = json.loads(built_model.to_json())

        # number of layers
        self.assertEqual(4, len(built_model.layers), "incorrect number of layer in model")

        # input
        self.assertEqual("InputLayer", j_model["config"]["layers"][0]["class_name"],
                         "input isn't specified as input layer")
        self.assertEqual([None, 10000], j_model["config"]["layers"][0]["config"]["batch_input_shape"],
                         "shape of input incorrect")
        self.assertEqual("float32", j_model["config"]["layers"][0]["config"]["dtype"],
                         "type of input incorrect")

        # layer types
        self.assertEqual("Dense", j_model["config"]["layers"][1]["class_name"],
                         "1st layer isn't specified as dense layer")
        self.assertEqual("Dense", j_model["config"]["layers"][2]["class_name"],
                         "2nd layer isn't specified as dense layer")
        self.assertEqual("Dense", j_model["config"]["layers"][3]["class_name"],
                         "3rd layer isn't specified as dense layer")

        # number of units
        self.assertEqual(64, j_model["config"]["layers"][1]["config"]["units"], "false number of units in 1st layer")
        self.assertEqual(64, j_model["config"]["layers"][2]["config"]["units"], "false number of units in 2nd layer")
        self.assertEqual(46, j_model["config"]["layers"][3]["config"]["units"], "false number of units in 3rd layer")

        # activation
        self.assertEqual("relu", j_model["config"]["layers"][1]["config"]["activation"],
                         "false activatio in 1st layer")
        self.assertEqual("relu", j_model["config"]["layers"][2]["config"]["activation"],
                         "false activation in 2nd layer")
        self.assertEqual("softmax", j_model["config"]["layers"][3]["config"]["activation"],
                         "false activation in 3rd layer")


    def test_evaluate(self):
        """
        Test evaluation
        """
        self.model.build()
        self.model.train(self.mock_train_set, self.mock_dev_set, 1, 512, save_model=False)
        loss, acc = self.model.evaluate(self.mock_test_set)

        # Loss
        self.assertIsNotNone(loss, "loss not computed")
        self.assertGreaterEqual(loss, 0., "loss is negativ")

        # Accuracy
        self.assertIsNotNone(acc, "accuracy not computed")
        self.assertGreaterEqual(acc, 0., "accuracy is negativ")
        self.assertLessEqual(acc, 1., "accuracy is greater than 1")


    def test_predict(self):
        """
        Test prediction
        """
        self.model.build()
        self.model.train(self.mock_train_set, self.mock_dev_set, 1, 512, save_model=False)
        test_output = self.model.predict(self.mock_input)

        # existence of prediction
        self.assertIsNotNone(test_output, "no prediction generated")

        # probabilities of each prediction must sum up to 1
        np.testing.assert_array_almost_equal(np.sum(test_output, axis=1),
                                             np.ones((self.mock_input.shape[0],)),
                                             err_msg="predicted probabilites don't sum to 1")

        # boundaries of prediction
        for i in test_output:
            for j in i:
                self.assertGreaterEqual(j, 0., "prediction out of boundary (is negativ)")
                self.assertLessEqual(j, 1., "prediction out of boundaries (is greater than 1)")


if __name__ == "__main__":
    unittest.main()
