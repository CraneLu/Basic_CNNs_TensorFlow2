import math, os
import tensorflow as tf
from configuration import *
from prepare_data import *

# get the dataset
train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False
base_model.summary()

def build_model():
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = build_model()
model.summary()

# 训练前评估模型（可选）
# loss0, accuracy0 = model.evaluate(valid_dataset.repeat(), steps=valid_count)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="log", histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'saved_model/training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
early_stopping_checkpoint = tf.keras.callbacks.EarlyStopping(patience=5)

steps_per_epoch = round(train_count) // BATCH_SIZE
validation_steps = round(valid_count) // BATCH_SIZE
history = model.fit(x=train_dataset.repeat(),
                              epochs=EPOCHS,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=valid_dataset.repeat(),
                              validation_steps=validation_steps,
                              callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopping_checkpoint])