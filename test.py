import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

datasets = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = datasets.load_data()

print(train_images.shape)  # (60000, 28, 28)
print(len(train_labels))  # 60000
print(train_labels)  # ([9, 0, 0, ..., 3, 0, 5], dtype=uint8)

print(test_images.shape)  # (10000, 28, 28)
print(len(test_labels))  # 10000
print(test_labels)  # ([9 2 1 ... 8 1 5], dtype=uint8)

