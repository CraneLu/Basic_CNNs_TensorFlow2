import tensorflow as tf
from configuration import *


base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
                                               include_top=False,
                                               weights='imagenet')