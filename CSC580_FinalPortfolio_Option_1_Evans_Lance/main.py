import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.layers import Input
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable TensorFlow debugging logs


# Define the file name to save the output
output_file_name = 'output.txt'

# Load the Data
# Loading the CIFAR-10 dataset
(X, y), (_, _) = cifar10.load_data()

# Select a single class from the dataset
X = X[y.flatten() == 5]

# Define parameters
# Define the input shape of the images
image_shape = (32, 32, 3)

# Define the latent space size
latent_dimensions = 100

# Define a utility function to build the generator
def build_generator():
    model = Sequential()

    # Building the input layer
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dimensions))
    model.add(Reshape((8, 8, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    # Generating the output image
    noise = Input(shape=(latent_dimensions,))
    image = model(noise)

    return Model(noise, image)

# Define a utility function to build the discriminator
def build_discriminator():
    # Building the convolutional neural network
    # to classify the images
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    # Building the output layer
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    image = Input(shape=image_shape)
    validity = model(image)

    return Model(image, validity)

# Define a utility function to display generated images
def display_images(generator, epoch):
    r, c = 4, 4
    noise = np.random.normal(0, 1, (r * c, latent_dimensions))
    generated_images = generator.predict(noise)

    # Scaling the generated images
    generated_images = 0.5 * generated_images + 0.5
    fig, axs = plt.subplots(r, c)
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(generated_images[count, :, :,])
            axs[i, j].axis('off')
            count += 1
    plt.savefig(f'generated_images_epoch_{epoch}.png')
    plt.close()

# Building and compiling the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Making the discriminator untrainable
# so that the generator can learn from fixed gradient
discriminator.trainable = False

# Build the generator
generator = build_generator()

# Defining the input for the generator and generating the images
z = Input(shape=(latent_dimensions,))
image = generator(z)

# Checking the validity of the generated image
valid = discriminator(image)

# Defining the combined model of the Generator and the Discriminator
combined_network = Model(z, valid)
combined_network.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training the network
num_epochs = 15000
batch_size = 128
display_interval = 100
losses = []

# Normalizing the input
X = (X / 127.5) - 1.0

# Defining the Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))


for epoch in range(num_epochs):
    sys.stdout = open(os.devnull, 'w')
    # Training the Discriminator
    # Selecting a random batch of images
    index = np.random.randint(0, X.shape[0], batch_size)
    images = X[index]
    # Sampling noise and generating a batch of new images
    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
    generated_images = generator.predict(noise)
    # Training the discriminator to detect real and fake images
    real_loss = discriminator.train_on_batch(images, valid)
    fake_loss = discriminator.train_on_batch(generated_images, fake)
    discriminator_loss = 0.5 * np.add(real_loss, fake_loss)
    # Calculating discriminator accuracy
    discriminator_acc = 0.5 * (real_loss[1] + fake_loss[1])
    # Training the Generator
    # Training the generator to generate images
    # which pass the authenticity test
    generator_loss = combined_network.train_on_batch(noise, valid)
    sys.stdout = sys.__stdout__

    # Tracking the progress
    if epoch % display_interval == 0:
        display_images(generator, epoch)
        # Printing progress
        print("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (
        epoch, discriminator_loss[0], 100 * discriminator_acc, generator_loss))
        losses.append((discriminator_loss[0], generator_loss))


# Plotting the losses
losses = np.array(losses)

# Plotting discriminator and generator loss
plt.figure(figsize=(15, 5))
plt.plot(losses.T[0], label='Discriminator loss')
plt.plot(losses.T[1], label='Generator loss')
plt.title("Training Losses")
plt.legend()
plt.savefig("training_losses.png")
plt.show()

# Plotting the images from the first epoch
display_images(generator, 0)

# Plotting the images from the last epoch
display_images(generator, num_epochs-1)
