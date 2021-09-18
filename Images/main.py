# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from Images.methods import plot_image, plot_value_array

print('---------------------------------------------------------------------------')
print('TensorFlow Version :', tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('Train Images :', train_images.shape)
print('Train Labels Len :', len(train_labels))
print('Train Labels :', train_labels)
print('Test Images :', test_images.shape)
print('Test Labels Len :', len(test_labels))
print('---------------------------------------------------------------------------')

# train_images = train_images / 255.0
#
# test_images = test_images / 255.0
#
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
#
# print('\nTest accuracy :', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print('---------------------------------------------------------------------------')
print('Test...')
print('\nPrediction List Of 0 :', predictions[0])
print('\nPrediction Label Of 0 :', np.argmax(predictions[0]))
print('\nVerify Prediction Label Of 0 :', test_labels[0])
print('\nPrediction Name Of 0 :', class_names[np.argmax(predictions[0])])
print('---------------------------------------------------------------------------')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Grab an image from the test dataset.
# img = test_images[1]

# Add the image to a batch where it's the only member.
# img = (np.expand_dims(img, 0))
#
# print(img.shape)
#
# predictions_single = probability_model.predict(img)
#
# print(predictions_single)
#
# plot_value_array(1, predictions_single[0], test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
# plt.show()
#
# print('\nPrediction of Image :', np.argmax(predictions_single[0]))
