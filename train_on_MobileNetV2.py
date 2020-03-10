import math, os
import tensorflow as tf
from tensorflow import keras
from configuration import *
from prepare_data import *

# get the dataset
train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
# datasets = tf.keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = datasets.load_data()

# init base model
base_model = keras.applications.MobileNetV2(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
base_model.summary()

# transfer model
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(12)
])
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# evaluate before training
loss0, accuracy0 = model.evaluate(valid_dataset.repeat())
# loss0, accuracy0 = model.evaluate(test_images, test_labels)

# # training
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir="log", histogram_freq=1)
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     'saved_model/training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
# early_stopping_checkpoint = keras.callbacks.EarlyStopping(patience=5)
# history = model.fit(x=train_dataset.repeat(),
#                               epochs=EPOCHS,
#                               steps_per_epoch=train_count // BATCH_SIZE,
#                               validation_data=valid_dataset.repeat(),
#                               validation_steps=valid_count // BATCH_SIZE,
#                               callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopping_checkpoint])