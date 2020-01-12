import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images/255.0
test_images = test_images/255.0

# Layers of the model, .Sequential = sequence of layers
model = keras.Sequential([
    # Flatten the data
    # Input Layer
    keras.layers.Flatten(input_shape=(28,28)),
    # Dense Layer #1 (fully connected layer)
    # "relu" - Rectified Linear Unit. It's a piecewise linear function
    # that will output the input directly if it's positive, output 0 if otherwise
    keras.layers.Dense(128, activation="relu"),
    # Dense Layer #2 (fully connected layer)
    # Softmax: Pick value for each neuron so the possibility add up to 1
    keras.layers.Dense(10, activation="softmax")
])

# Pass in parameters
model.compile(optimizer = "adam", Loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# Train the model
# Epochs, Number of complete passes through the training data-set
model.fit(train_images, train_labels, epochs=5)

# Evaluate the accuracy and loss
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Print out the accuracy of the model
print("Tested Accuracy", test_acc)