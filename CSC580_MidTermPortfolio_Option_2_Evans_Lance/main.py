from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# from tensorflow.keras import layers
# print(tf.__version__)
# Download dataset from UCI Machine Learning Repository using keras
dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
)
# print(dataset_path)
# Import dataset using pandas
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]
raw_dataset = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True,
)
dataset = raw_dataset.copy()

# print(dataset.tail())
# print(dataset.isna().sum())
# Drop rows with missing values
dataset = dataset.dropna()

# Split dataset into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect data
sns.pairplot(
    train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde"
)
# plt.show()

# Inspect overall statistics
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
# print(train_stats)

# split features from labels


# Separate target value from features
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")


# Normalize data
def norm(x):
    return (x - train_stats["mean"]) / train_stats["std"]


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# Build model
def build_model():
    model = keras.Sequential(
        [
            keras.layers.Dense(
                64, activation="relu", input_shape=[len(train_dataset.keys())]
            ),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ]
    )

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])

    return model


model = build_model()


# Inspect model
#  print(model.summary())

# Try out model
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)

# print(example_result)


# Train model
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


EPOCHS = 1000
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
early_history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, tfdocs.modeling.EpochDots()],
)


history = model.fit(
   normed_train_data,
   train_labels,
   epochs=EPOCHS,
   verbose=0,
   validation_split=0.2,
   callbacks=[tfdocs.modeling.EpochDots()],
)
# Visualize training progress
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
print(hist.tail())
#
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({"Basic": history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel("MAE [MPG]")
plt.show()
#
plotter.plot({"Basic": history}, metric="mse")
plt.ylim([0, 20])
plt.ylabel("MSE [MPG^2]")
plt.show()

plotter.plot({"Early Stopping": early_history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel("MAE [MPG]")
plt.show()

# Evaluate model
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# Make predictions
test_predictions = model.predict(normed_test_data).flatten()
# print(test_predictions)
a = plt.axes(aspect="equal")
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

# Error distribution
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
