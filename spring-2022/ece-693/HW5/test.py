# %% [markdown]
# <a href="https://colab.research.google.com/github/mschrader15/homework/blob/main/spring-2022/ece-693/HW5.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ## **Introduction to homework**
# This homework explores conditional GAN that produces images conditioned on class labels. This homework is based on the example by Sayak Paul available at https://keras.io/examples/generative/conditional_gan/

# %% [markdown]
# "Generative Adversarial Networks (GANs) let us generate novel image data, video data,
# or audio data from a random input. Typically, the random input is sampled
# from a normal distribution, before going through a series of transformations that turn
# it into something plausible (image, video, audio, etc.).
# 
# However, a simple [DCGAN](https://arxiv.org/abs/1511.06434) doesn't let us control
# the appearance (e.g. class) of the samples we're generating. For instance,
# with a GAN that generates MNIST handwritten digits, a simple DCGAN wouldn't let us
# choose the class of digits we're generating.
# To be able to control what we generate, we need to _condition_ the GAN output
# on a semantic input, such as the class of an image."
# 
# In this homework, you'll train a **Conditional GAN** that can generate food images of a given class.
# 
# The original MNIST example is based on these references:
# 
# * [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
# * [Lecture on Conditional Generation from Coursera](https://www.coursera.org/lecture/build-basic-generative-adversarial-networks-gans/conditional-generation-inputs-2OPrG)
# 
# For a description of GANs, you can refer to the "Generative adversarial networks"
# section of
# [this resource](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-12/r-3/232).
# 
# This homework requires TensorFlow 2.5 or higher, as well as TensorFlow Docs, which can be
# installed using the following command:
# %% [markdown]
# ## Imports

# %%
import os
import shutil
import time
import glob
import random
from PIL import Image

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio


# %% [markdown]
# ## Constants and hyperparameters

# %%
batch_size = 64

# RGB
num_channels = 3

# cats and dogs
num_classes = 2

image_size = 540
latent_dim = 128

IMG_SIZE = (image_size, image_size)

# %% [markdown]
# ## Loading the dataset and preprocessing it
# 
# 

# %% [markdown]
# 
# ### **Assignment dataset**
# 
# Note that below is the original code for loading the MNIST dataset. In this homework you could use **one** of the following:
# 1. CIFAR10 dataset. This dataset is included with Keras, See https://keras.io/api/datasets/cifar10/
# 2. Cats and Dogs dataset. This dataset is available from Kaggle. The textbook has examples of working with this dataset in the image classification section. https://www.kaggle.com/competitions/dogs-vs-cats/data
# 3. Food101 dataset, also available from Kaggle. https://www.kaggle.com/datasets/kmader/food41
# 
# Regardless of the dataset you choose, a couple of important considerations:
# - Unlike the MNIST example, these are color images and the GAN should be able to produce color images.
# - Keep the image resolution and number of classes manageable, otherwise the network will take too long or will not properly train.
# 
# **Modify the code below to load the dataset of your choice**
# 
# Please either use wget links or include the data in the same directory as your code.
# Note that I'll not be able to access files mounted on your Google drive, so the data should be local to the runtime instance!!! You could use wget or gdown to download required files.

# %% [markdown]
# ### Store the File Paths

# %%
ROOT = os.path.abspath("")
TEST_DATA = os.path.join(ROOT, 'data', 'test1')
TRAIN_DATA = os.path.join(ROOT, 'data', 'train')

# %% [markdown]
# ### Split the Data into Test, Train and Validate

# %%
def make_subset(data_dir, subset_name, start_index, end_index) -> str:
    for category in ("cat", "dog"):
        new_dir = os.path.join(data_dir, subset_name, category)
        try:
            os.makedirs(new_dir)
            fnames = [(f"{category}.{i}.jpg", f"{i}-{category}.jpg") for i in range(start_index, end_index)]
            for fname in fnames:
                shutil.copyfile(src=os.path.join(data_dir, fname[0]), dst=os.path.join(new_dir, fname[1]))
        except FileExistsError:
            break
    return os.path.join(data_dir, subset_name)

# %%
_, _, files = next(os.walk(TRAIN_DATA))
print(num_images:=len(files))

# %% [markdown]
# #### Loading the Datasets

# %%
# DATA_PERCENTAGE = 10

# # %%
# file_list = glob.glob(os.path.join(TRAIN_DATA, '*.jpg'))[:int(DATA_PERCENTAGE * num_images)]


# # %%
# image_data = np.array([np.array(Image.open(fname)) for fname in file_list])

# %%

test_dataset = image_dataset_from_directory(
    make_subset(TRAIN_DATA, "train", start_index=0, end_index=2500),
    image_size=IMG_SIZE,
    batch_size=batch_size,
)
# validation_dataset = image_dataset_from_directory(
#     make_subset(TRAIN_DATA, "validation", start_index=1000, end_index=1500),
#     image_size=IMG_SIZE,
#     batch_size=batch_size,
# )
# test_dataset = image_dataset_from_directory(
#     make_subset(TRAIN_DATA, "test", start_index=1500, end_index=2500),
#     image_size=IMG_SIZE,
#     batch_size=batch_size,
# )


# # %%
# for d, i in dataset.take(1):
#     break

# %%
# test_dataset

# # %%
# dataset

# %% [markdown]
# ### Plotting Image to Ensure Everything OK

# %%
import matplotlib.pyplot as plt

# plotting the data augmentation and the color distortion
plt.figure(figsize=(10, 10))
for images, _ in test_dataset.take(1):
        # augmented_images = data_augementation()(images)
        # ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")

# %%
# We'll use all the available examples from both the training and test
# sets.
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# all_digits = np.concatenate([x_train, x_test])
# all_labels = np.concatenate([y_train, y_test])

# # Scale the pixel values to [0, 1] range, add a channel dimension to
# # the images, and one-hot encode the labels.
# all_digits = all_digits.astype("float32") / 255.0
# all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
# all_labels = keras.utils.to_categorical(all_labels, 10)

# # Create tf.data.Dataset.
# dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
# dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# print(f"Shape of training images: {all_digits.shape}")
# print(f"Shape of training labels: {all_labels.shape}")

# %% [markdown]
# ## Calculating the number of input channel for the generator and discriminator
# 
# In a regular (unconditional) GAN, we start by sampling noise (of some fixed
# dimension) from a normal distribution. In our case, we also need to account
# for the class labels. We will have to add the number of classes to
# the input channels of the generator (noise input) as well as the discriminator
# (generated image input).

# %%
generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

# %% [markdown]
# ## Creating the discriminator and generator
# 
# The model definitions (`discriminator`, `generator`, and `ConditionalGAN`) have been
# adapted from [this example](https://keras.io/guides/customizing_what_happens_in_fit/).
# 
# Note that you need to adjust these models for work with color images.

# %%
# Create the discriminator.
discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((28, 28, discriminator_in_channels)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Create the generator.
generator = keras.Sequential(
    [
        keras.layers.InputLayer((generator_in_channels,)),
        # We want to generate 128 + num_classes coefficients to reshape into a
        # 7x7x(128 + num_classes) map.
        layers.Dense(7 * 7 * generator_in_channels),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, generator_in_channels)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)

# %%
tf.keras.utils.plot_model(discriminator,show_shapes=True)

# %%
tf.keras.utils.plot_model(generator,show_shapes=True)

# %% [markdown]
# ## Creating a `ConditionalGAN` model

# %%

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        print(one_hot_labels)
        image_one_hot_labels = one_hot_labels[:, ]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }


# %% [markdown]
# ## Training the Conditional GAN
# 
# **Do not forget to save the model after training and include the model with the submission**

# %%
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)
cond_gan.run_eagerly = True
cond_gan.fit(test_dataset, epochs=20)

# %% [markdown]
# ## Interpolating between classes with the trained generator
# 
# **Load the model from the file, generate images**
# 
# Note that the class choice may be set the same for start_class and end_class. In this case generate 1 image of the target class.

# %%
# We first extract the trained generator from our Conditiona GAN.
trained_gen = cond_gan.generator

# Choose the number of intermediate images that would be generated in
# between the interpolation + 2 (start and last images).
num_interpolation = 9  # @param {type:"integer"}

# Sample noise for the interpolation.
interpolation_noise = tf.random.normal(shape=(1, latent_dim))
interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, latent_dim))


def interpolate_class(first_number, second_number):
    # Convert the start and end labels to one-hot encoded vectors.
    first_label = keras.utils.to_categorical([first_number], num_classes)
    second_label = keras.utils.to_categorical([second_number], num_classes)
    first_label = tf.cast(first_label, tf.float32)
    second_label = tf.cast(second_label, tf.float32)

    # Calculate the interpolation vector between the two labels.
    percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = tf.cast(percent_second_label, tf.float32)
    interpolation_labels = (
        first_label * (1 - percent_second_label) + second_label * percent_second_label
    )

    # Combine the noise and the labels and run inference with the generator.
    noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
    fake = trained_gen.predict(noise_and_labels)
    return fake


start_class = 1  # @param {type:"slider", min:0, max:9, step:1}
end_class = 7  # @param {type:"slider", min:0, max:9, step:1}

fake_images = interpolate_class(start_class, end_class)

# %% [markdown]
# Here, we first sample noise from a normal distribution and then we repeat that for
# `num_interpolation` times and reshape the result accordingly.
# We then distribute it uniformly for `num_interpolation`
# with the label indentities being present in some proportion.

# %%
fake_images *= 255.0
converted_images = fake_images.astype(np.uint8)
converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
imageio.mimsave("animation.gif", converted_images, fps=1)
embed.embed_file("animation.gif")

# %% [markdown]
# We can further improve the performance of this model with recipes like
# [WGAN-GP](https://keras.io/examples/generative/wgan_gp).
# Conditional generation is also widely used in many modern image generation architectures like
# [VQ-GANs](https://arxiv.org/abs/2012.09841), [DALL-E](https://openai.com/blog/dall-e/),
# etc.
# 
# You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/conditional-gan) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/conditional-GAN).

# %% [markdown]
# # **Grading**
# 
# *   (10 pts) Report quality / submission requirement followed (see class policy)
# *   (90 pts) Correctness of the implementation. The basis is ability of the network to generate coherent shapes that may remotely resemble instances of the target class (as opposed to random noise).
# *   (10 pts bonus) Explore how **one** of the following affects the realism of image generattion: 1) number of classes; 2) size of the latent dimension; 3) image resolution; 4) increasing the depth / layer type (e.g. implementing residual connections) of the discriminator
# 
# 

# %% [markdown]
# # **Submission**
# Using Blackboard, submit the .ipynb file, the dataset files, saved models and the report in a compressed folder as defined by the class policy.
# 
# Note that I will not have access to your Google drive. All files must be local or downloadable by wget/gdown. 


