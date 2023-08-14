import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = os.path.join('', 'train')
train_normal = os.path.join(data_dir, 'normal')
train_sick = os.path.join(data_dir, 'pneumonia')


X = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode='binary',
    labels='inferred',
    color_mode='grayscale',
    class_names=['normal', 'pneumonia'],
    shuffle=True,
    image_size=(100, 100),
    interpolation='nearest',
    crop_to_aspect_ratio=False,
    validation_split=0.8,
    subset='training',
    seed=123
)
X_val = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode='binary',
    labels='inferred',
    color_mode='grayscale',
    class_names=['normal', 'pneumonia'],
    shuffle=True,
    image_size=(100, 100),
    interpolation='nearest',
    crop_to_aspect_ratio=False,
    validation_split=0.2,
    subset='validation',
    seed=123

)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(100, 100, 1), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, batch_input_shape = (32,) + tuple([100, 100]), activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
epoc = [i for i in range(5)]
hist = model.fit(X, steps_per_epoch=5, epochs=5, validation_data=(X_val), validation_batch_size=32, validation_steps=5)
fig = plt.figure(figsize=(14, 10))
fig, ax = plt.subplots(1, 2, squeeze=False)
ax[0,0].plot(epoc, hist.history['loss'], 'r')
ax[0,0].plot(epoc, hist.history['accuracy'], 'b')
ax[0, 1].plot(epoc, hist.history['val_loss'], 'r')
ax[0, 1].plot(epoc, hist.history['val_accuracy'], 'b')

plt.show()
