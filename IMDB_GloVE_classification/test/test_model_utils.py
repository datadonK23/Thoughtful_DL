import unittest
import os, shutil, tempfile
from keras import models, layers

from model_utils import build_model, save_model, load_model


class ModelUtilsTests(unittest.TestCase):

    def setUp(self):
        self.test_model_dir = tempfile.mkdtemp()
        self.mock_model = models.Sequential([layers.Dense(1, input_shape=(256,)),
                                             layers.Activation("softmax")]) # for IO tests


    def tearDown(self):
        shutil.rmtree(self.test_model_dir)


    def test_build_model(self):
        """
        Test architecture of model
        """
        pass #FIXME


    def test_save_model(self):
        """
        Test model saving
        """
        mock_model = self.mock_model
        save_model(mock_model, self.test_model_dir)
        files = os.listdir(self.test_model_dir)
        if files:
            test_ext = os.path.splitext(files[0])[1]

        # Saves a model
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
            model_name = files[0]  # hardcoded, take first model from model_dir
            loaded_model = load_model(self.test_model_dir, model_name)

        # loaded model exists
        self.assertTrue(loaded_model, "no model loaded")

