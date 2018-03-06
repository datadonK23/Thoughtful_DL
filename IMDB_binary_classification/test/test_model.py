import unittest
import numpy as np
import json
import os, shutil, tempfile

from model import Model


class ModelTests(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        self.test_model_dir = tempfile.mkdtemp()
        self.mock_train_set = (np.zeros((1, 10000)), np.zeros((1,)))
        self.mock_dev_set = self.mock_train_set
        self.mock_test_set = self.mock_train_set
        self.mock_input = np.random.randint(2, size=10000).reshape(1, 10000)


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
        self.assertEqual(16, j_model["config"]["layers"][1]["config"]["units"], "false number of units in 1st layer")
        self.assertEqual(16, j_model["config"]["layers"][2]["config"]["units"], "false number of units in 2nd layer")
        self.assertEqual(1, j_model["config"]["layers"][3]["config"]["units"], "false number of units in 3rd layer")

        # activation
        self.assertEqual("relu", j_model["config"]["layers"][1]["config"]["activation"],
                         "false activatio in 1st layer")
        self.assertEqual("relu", j_model["config"]["layers"][2]["config"]["activation"],
                         "false activation in 2nd layer")
        self.assertEqual("sigmoid", j_model["config"]["layers"][3]["config"]["activation"],
                         "false activation in 3rd layer")


    def test_train(self):
        """
        Test training step
        """
        self.model.build()
        trained_model = self.model.train(self.mock_train_set, self.mock_dev_set,
                                         epochs=1, batch_size=512)

        self.assertIsNotNone(trained_model[1], "no model trained")
        self.assertIsNotNone(trained_model[0], "history dict doesn't exist")


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


    def test_save(self):
        """
        Test model saving
        """
        model = self.model.build()
        self.model.save(model, self.test_model_dir)
        files = os.listdir(self.test_model_dir)
        if files:
            test_ext = os.path.splitext(files[0])[1]

        # Saves a model
        self.assertTrue(files, "no model saved")
        self.assertEqual(".h5", test_ext, "model not saved as '.h5'")


    def test_load(self):
        """
        Test model loading
        """
        model = self.model.build()
        loaded_model = None
        self.model.save(model, self.test_model_dir)
        files = os.listdir(self.test_model_dir)
        if files:
            model_name = files[0] # hardcoded, take first model from model_dir
            loaded_model = self.model.load(self.test_model_dir, model_name)

        # loaded model exists
        self.assertTrue(loaded_model, "no model loaded")


    def test_predict(self):
        """
        Test prediction
        """
        self.model.build()
        self.model.train(self.mock_train_set, self.mock_dev_set, 1, 512, save_model=False)
        test_output = self.model.predict(self.mock_input)

        # existence of prediction
        self.assertIsNotNone(test_output, "no prediction generated, without given model name")

        # boundaries of prediction
        self.assertGreaterEqual(test_output, 0., "prediction out of boundaries (is negativ)")
        self.assertLessEqual(test_output, 1., "prediction out of boundaries (is greater than 1)")

        # Tests with specific model name as input
        files = os.listdir(self.test_model_dir)
        if files:
            model_name = files[0]  # hardcoded, take first model from model_dir
            test_output_from_model = self.model.predict(self.mock_input, model_name)

            #existence
            self.assertIsNotNone(test_output_from_model, "no prediction generated, with given model name")

            # boundaries of prediction
            self.assertGreaterEqual(test_output_from_model, 0.,
                                    "prediction from loaded model out of boundaries (is negativ)")
            self.assertLessEqual(test_output_from_model, 1.,
                                 "prediction from loaded model out of boundaries (is greater than 1)")


if __name__ == "__main__":
    unittest.main()
