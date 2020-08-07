
import tensorflow as tf
import keras
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist
from keras import Sequential, layers

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
               'Bag', 'Ankle boot']


def plotOne(x):
    plt.figure()
    plt.imshow(train_images[x])
    plt.colorbar()
    plt.grid(False)
    plt.show()


# plotOne(1)

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
print(train_images.shape, train_labels.shape)
print(train_images[0])
print(train_labels[:10])
model1 = Sequential([layers.Flatten(input_shape=(28, 28)),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(64, activation='relu'),
                     layers.Dense(10, activation='softmax')])
print(model1.summary())
model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model1.fit(train_images, train_labels, epochs=20)
test_loss, test_accuracy = model1.evaluate(test_images, test_labels, verbose=2)
print("test accuracy:", test_accuracy)