# machine_learning_02

Udacity course: https://classroom.udacity.com/courses/ud187

## Lesson 3: Your First Model - Fashion MNIST
We use the MNIST dataset, which is 70000 images of various clothing items.
* Each image is 28x28 pixels, or 784 pixels.
* Each image is converted to a 784-byte array, and TensorFlow is given a vector of 784-byte objects.

```
# Layer for input images
tf.keras.layers.Flatten(input_shape=(28, 28, 1))

# Dense layer for comparison.  Will learn ReLU activation later.
tf.keras.layers.Dense(128, activation=tf.nn.relu)

# Use softmax to generate the probability that a given item fits within a given category.
tf.keras.layers.Dense(10, activation=tf.nn.softmax)
```

### ReLU: Rectified Linear Unit
This is an "activation function" that allows models to predict **non-linear** problems.

For input `x`, output `y`:
```
if x < 0:
  y = 0
else:
  y = x
```

ReLU is useful for:

1. Models that have an interaction effect between variables.  For example, if we are predicting the probability of diabetes given a person's height and weight, the impact of weight is dependent on height.

2. Models that have non-linear effects.  This "activation function" is essentially allowing us to weigh a specific input variable in a non-linear fashion. When the input variable's value is higher, it carries even more weight.

So especially when we use ReLU with a dense network and multiple layers, we can combine them with different slopes and biases to generate complex results.

### Splitting data between training and testing
This allows us to ensure that the model is genuinely *predicting* values, rather than just having a sufficient mapping of input to output.

#### Coding MNIST
See [Udacity: Classifying Images of Clothing](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l03c01_classifying_images_of_clothing.ipynb).

Uses TensorFlow Datasets, which simplifies downloading and accessing various datasets.  It also includes a collection of sample datasets, including MNIST: 
```
pip install tensorflow_datasets
```

With the environment setup, here is the code:
```
import tensorflow as tf

# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# The dataset includes metadata, a training dataset, AND a test dataset.
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# See the different classes of clothing.
class_names = metadata.features['label'].names
print("Class names: {}".format(class_names))

def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

# Instead of values being in the range of [0,255], they need to be normalized to [0,1].
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk.
# Caching will keep them in memory, making training faster.
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()

# Generate and compile the model.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),   # input layer
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # hidden layer
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # output layer
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model.
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

# Check the accuracy on the test dataset.
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)
```

For further examples (including plotting results), see the CoLab page.

**Next steps**: Lesson 4: Introduction to CNNs

## Sample Code
I'm revisiting the model that I did with machine_learning_01.  That was obviously flawed, since:
* I created a polynomial model for an exponential calculation.
* I didn't bother to create a large training sample set.
* I didn't create a testing sample set.

I'm going to correct all of those here.  I'm also going to compare the accuracy of a model that uses the default activation (a polynomial model), versus a model that uses ReLU activation (a non-linear model).

### System Requirements

* Python 3.7
* Pip 20.1+
* Poetry 1.1.4+:
  * `pip install poetry`
* (*On Windows 7 or later*) [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads)
  * Required by tensorflow.  [Details](https://www.tensorflow.org/install/pip#system-requirements).

### Project Setup
To setup the virtual environment, run:
  > `poetry install`

To execute the sample code, run:
  > `poetry run main`

### Project Results