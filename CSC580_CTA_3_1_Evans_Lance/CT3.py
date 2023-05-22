import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

np.random.seed(101)
tf.compat.v1.set_random_seed(101)

# Generate random linear data
# There will be 50 data points ranging from 0 to 50
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

# Adding noise to the random linear data
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

n = len(x)  # Number of data points

# Plot of Training Data
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training Data")
plt.show()

x_train = np.array(x).reshape(-1, 1)
y_train = np.array(y).reshape(-1, 1)

weights = tf.Variable(np.random.randn(), name="weights", dtype=tf.float64)
bias = tf.Variable(np.random.randn(), name="bias", dtype=tf.float64)

learning_rate = 0.001
training_epochs = 5000

# Hypothesis
y_pred = tf.add(tf.multiply(x_train, weights), bias)


# Mean Squared Error Cost Function
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


cost = mean_squared_error(y_train, y_pred)

# Gradient Descent Optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Global Variables Initializer
init = tf.compat.v1.global_variables_initializer()

# Starting the Tensorflow Session
with tf.compat.v1.Session() as sess:
    # Initializing the Variables
    sess.run(init)

    # Iterating through all the epochs
    for epoch in range(training_epochs):
        # Feeding each data point into the optimizer using Feed Dictionary
        sess.run(optimizer)

        # Displaying the result after every 50 epochs
        if (epoch + 1) % 50 == 0:
            # Calculating the cost a every epoch
            c = sess.run(cost)
            print(
                "Epoch",
                (epoch + 1),
                ": cost =",
                c,
                "weights =",
                sess.run(weights),
                "bias =",
                sess.run(bias),
            )

    # Storing necessary values to be used outside the Session
    training_cost = sess.run(cost)
    weight = sess.run(weights)
    bias = sess.run(bias)

# Calculating the predictions
predictions = weight * x + bias
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, "\n")

# Plotting the Results
plt.plot(x, y, "ro", label="Original data")
plt.plot(x, predictions, label="Fitted line")
plt.title("Linear Regression Result")
plt.legend()
plt.show()

