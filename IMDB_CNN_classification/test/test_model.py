import unittest
import os, json, pickle, shutil, tempfile
from keras import models, layers

from model import build_model, train_model, evaluate_model, plot_model, save_model, load_model
from preprocessing import load_data, pad_texts


class ModelTests(unittest.TestCase):

    def setUp(self):
        self.test_model_dir = tempfile.mkdtemp()
        self.test_plots_dir = tempfile.mkdtemp()
        self.mock_model = models.Sequential([layers.Dense(1, input_shape=(256,)),
                                             layers.Activation("softmax")]) # for IO tests


    def tearDown(self):
        shutil.rmtree(self.test_model_dir)
        shutil.rmtree(self.test_plots_dir)

    @unittest.skip("FIXME")
    def test_build_model(self):
        """
        Test architecture of model
        """
        mock_embedding_matrix = pickle.load(open("models/mock_glove.6B/mock_embedding_matrix.p", "rb"))
        mock_model = build_model(mock_embedding_matrix)
        j_model = json.loads(mock_model.to_json())

        # model type
        self.assertEqual("Sequential", j_model["class_name"], "Built model is not a sequential model")

        # number of layers
        self.assertEqual(4, len(mock_model.layers), "incorrect number of layer in model")

        # embedding input
        self.assertEqual("Embedding", j_model["config"][0]["class_name"],
                         "First layer isn't an embedding layer")
        self.assertEqual([None, 100], j_model["config"][0]["config"]["batch_input_shape"],
                         "shape of embedding input incorrect")
        self.assertEqual("float32", j_model["config"][0]["config"]["dtype"],
                         "type of embedding input incorrect")
        self.assertEqual(False, j_model["config"][0]["config"]["trainable"],
                         "embedding input is trainable")

        # Flatten layer
        self.assertEqual("Flatten", j_model["config"][1]["class_name"],
                         "Second layer isn't a flatten layer")
        self.assertEqual(True, j_model["config"][1]["config"]["trainable"],
                         "flatten layer is not trainable")

        # layer types
        self.assertEqual("Dense", j_model["config"][2]["class_name"],
                         "3nd layer isn't specified as dense layer")
        self.assertEqual("Dense", j_model["config"][3]["class_name"],
                         "4th layer isn't specified as dense layer")

        # number of units
        self.assertEqual(32, j_model["config"][2]["config"]["units"], "false number of units in 3rd layer")
        self.assertEqual(1, j_model["config"][3]["config"]["units"], "false number of units in 4th layer")

        # trainability
        self.assertEqual(True, j_model["config"][2]["config"]["trainable"],
                         "3rd layer is not trainable")
        self.assertEqual(True, j_model["config"][3]["config"]["trainable"],
                         "4th layer is not trainable")

        # activations
        self.assertEqual("relu", j_model["config"][2]["config"]["activation"],
                         "false activation in 3rd layer")
        self.assertEqual("sigmoid", j_model["config"][3]["config"]["activation"],
                         "false activation in 4th layer")

    @unittest.skip("FIXME")
    def test_train_model(self):
        """
        Test if function returns trained model
        """
        texts, labels = preprocess_labels(data_dir_path="data/mock_aclImdb", dataset="train")
        vectorized_texts, word_index = tokenize_data(texts)
        mock_X_train, mock_y_train, mock_X_val, mock_y_val = split_data(vectorized_texts, labels)

        mock_embedding_matrix = pickle.load(open("models/mock_glove.6B/mock_embedding_matrix.p", "rb"))
        mock_model = build_model(mock_embedding_matrix)

        mock_trained_model = train_model(mock_model, (mock_X_train, mock_y_train), (mock_X_val, mock_y_val))

        self.assertIsNotNone(mock_trained_model[1], "no model trained")
        self.assertIsNotNone(mock_trained_model[0], "history dict doesn't exist")

    @unittest.skip("FIXME")
    def test_evaluate_model(self):
        """
        Test boundaries of loss and accuracy
        """
        texts, labels = preprocess_labels(data_dir_path="data/mock_aclImdb", dataset="test")
        vectorized_texts, word_index = tokenize_data(texts)
        mock_test_set = (vectorized_texts, labels)
        mock_trained_model = load_model("models/", "mock_trained_model.h5")

        loss, acc = evaluate_model(mock_trained_model, mock_test_set)

        # Loss
        self.assertIsNotNone(loss, "loss not computed")
        self.assertGreaterEqual(loss, 0., "loss is negativ")

        # Accuracy
        self.assertIsNotNone(acc, "accuracy not computed")
        self.assertGreaterEqual(acc, 0., "accuracy is negativ")
        self.assertLessEqual(acc, 1., "accuracy is greater than 1")

    @unittest.skip("FIXME")
    def test_plot_model(self):
        """
        Test if plotted model is saved to disk
        """
        mock_model = self.mock_model
        plot_model(mock_model, self.test_plots_dir)
        files = os.listdir(self.test_plots_dir)
        if files:
            test_ext = os.path.splitext(files[0])[1]

        # Saves a models
        self.assertTrue(files, "no model plot saved")
        self.assertEqual(".png", test_ext, "model plot not saved as '.png'")


    def test_save_model(self):
        """
        Test model saving
        """
        mock_model = self.mock_model
        save_model(mock_model, self.test_model_dir)
        files = os.listdir(self.test_model_dir)
        if files:
            test_ext = os.path.splitext(files[0])[1]

        # Saves a models
        self.assertTrue(files, "no model saved")
        self.assertEqual(".h5", test_ext, "model not saved as '.h5'")


    def test_load_model(self):
        """
        Test model loading
        """
        model = self.mock_model
        loaded_model = None
        save_model(model, self.test_model_dir)
        files = os.listdir(self.test_model_dir)
        if files:
            model_name = files[0]  # hardcoded, take first models from model_dir
            loaded_model = load_model(self.test_model_dir, model_name)

        # loaded models exists
        self.assertTrue(loaded_model, "no model loaded")
