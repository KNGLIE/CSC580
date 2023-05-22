import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score
import datetime

np.random.seed(456)
tf.random.set_seed(456)

_, (train, valid, test), _ = dc.molnet.load_tox21()

train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

d = 1024
n_hidden = 50
learning_rate = 0.001
n_epochs = 10
batch_size = 100

inputs = tf.keras.Input(shape=(d,))
x_hidden = tf.keras.layers.Dense(n_hidden, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1)(x_hidden)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "/tmp/fcnet-tox21/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_X, train_y, batch_size=batch_size, epochs=n_epochs,
                    validation_data=(valid_X, valid_y), verbose=1,
                    callbacks=[tensorboard_callback])

valid_y_pred = model.predict(valid_X)
valid_y_pred = np.round(tf.sigmoid(valid_y_pred))

valid_acc = accuracy_score(valid_y, valid_y_pred)
print("Validation Accuracy: %f" % valid_acc)

plt.plot(history.history['loss'])
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
