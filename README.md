# Thoughtful Deep Learning with Python
Deep Learning with Python and Keras, the thoughtful way. Testing and validating Keras projects.

Original examples are taken from the book [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by Francois Chollet. These are enhanced with methods that are highly influenced by the book [Thoughtful Machine Learning with Python](http://shop.oreilly.com/product/0636920039082.do) by Matthew Kirk.


## Content
Project name and purpose of example:

* IMDB_binary_classification - Unittest all the things (incl. anti-pattern of testing the framework)
* Reuters_multiclass_classification - Plot training history and evaluate metrices of final model
* HousePrices_regression - Imperative project setup and K-fold validation
* IMDB_GloVe_classification - Functional project setup, text tokenizations, plot model informations, use pretrained embeddings
* IMDB_CNN_classification - Callbacks and Tensorboard for monitoring training

For more details, see `readme.md` in each projects main directory.


## Requirements
Run `conda env create -f environment.yml` to create an environment with all dependencies. Afterwards run `conda activate DeepLearning` to activate it.


## License
MIT License
