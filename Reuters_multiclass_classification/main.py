"""
    Multiclass Classification with newswire dataset
    Data: Reuters dataset (imported with Keras), orig 1986
    Input: 11,228 newswires, annotated with 46 different topics
    Model: Sequential, developed with Keras
    Output: Probabilities to which topic the newswire belongs
"""
from keras import backend

from model import *
from preprocessing import *


def main():
    """
    Main Loop

    :return: -
    """
    backend.clear_session()

    (train_data, train_labels), (dev_data, dev_labels),  (test_data, test_labels) = Data().train_dev_test

    # Model exploration
    model_explore = Model()
    history, _ = model_explore.train((train_data, train_labels), (dev_data, dev_labels), epochs=20)
    model_explore.plot_history(history)

    # Training final model
    model_final = Model()
    history, _ = model_final.train((train_data, train_labels), (dev_data, dev_labels), epochs=9)
    loss, acc = model_final.evaluate((test_data, test_labels))

    # Peformance evaluation
    bayes_error = 0.05
    training_error = 1 - history["acc"][-1]
    dev_error = 1 - acc
    avoidable_bias = training_error - bayes_error
    variance = dev_error - training_error
    print("Model Performance: ")
    print("Loss: " + str(loss))
    print("Accuracy: " + str(acc))
    print("Avoidable bias: " + str(avoidable_bias))
    print("Variance: " + str(variance))
    if avoidable_bias >= variance:
        print("You have a problem with avoidable bias. Tactics:")
        print("Bigger model; Train longer/better optimization algo; NN architecture/Hyperparameter search")
    else:
        print("You have a problem with variance. Tactics:")
        print("More data; Regularization; NN architecture/Hyperparameter search")

    # Model constrains
    assert acc >= 0.77, "Accuracy of final model is too low"
    assert avoidable_bias < 0.1, "Avoidable bias is too high, use tactics to fix this"
    assert variance < 0.15, "Variance is too high, use tactics to fix this"

    # Predict
    input = np.random.randint(2, size=20000).reshape(2, 10000)
    predictions = model_final.predict(input)
    print()
    print("Predictions:")
    for i, prediction in enumerate(predictions):
        print("Prediction for Newswire {}: Topic {}".format(i, np.argmax(prediction)))


if __name__ == "__main__":
    main()