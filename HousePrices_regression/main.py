#!/usr/bin/python
# encoding: utf-8
"""
    Regression to predict house prices
    Data: Boston Housing Price dataset (imported with Keras), orig mid 1970s
    Input: 506 examples, contain each 13 attributes of houses
    Model: Sequential, developed with Keras
    Output: Predict median price of homes in given Boston suburb
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend, models, layers
from keras.datasets import boston_housing

backend.clear_session()

# Load data
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()  # notice no dev set

# Normalization
# notice only training data is used for computation
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    """
    Builds sequential model for repetitive calls

    :return: model
    """
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

    return model

# K-fold validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print("Fold #", i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[: i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples], train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )
    # Build and fit model for each fold
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history["val_mean_absolute_error"]
    all_mae_histories.append(mae_history)

# Plot avg mae history
avg_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

def smooth_curve(points, factor=0.9):
    """
    Replace point with exponential moving average of previous points

    :param points: data points from model history
    :param factor: smoothing factor [0,1]
    :return: [smoothed_points]
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(avg_mae_history[10:]) # notice first 10 points are removed

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Val MAE")
plt.show()

# Final model

# Train
model = build_model()
model.fit(train_data, train_targets,
          epochs=35, batch_size=16, verbose=0)

# Save
model_dir = os.path.join(os.path.dirname(__file__), "model/")
f_name = model.name + ".h5"
model.save(os.path.join(model_dir, f_name))

# Evaluate
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
assert test_mae_score <= 3., "MAE is more than 3,000$"
print("MAE of model: ", test_mae_score)
