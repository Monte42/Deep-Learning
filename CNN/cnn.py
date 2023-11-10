# IMPORTS
import tensorflow as tf
import numpy as np
import os

# =====================
# DATA PRE_PROCESSING
# =====================

train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/training_set',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(64,64),
    batch_size=32
)
# class_names = train_ds.class_names
val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/test_set',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(64,64),
    batch_size=32
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



# =================================
# BUILD CONVOLUTION NEURAL NETWORK
# =================================
# INITIALIZE CONVOLUTION NEURAL NETWORK
cnn = tf.keras.models.Sequential()

# RESCALING
cnn.add(tf.keras.layers.Rescaling(1./255))

# DATA AUGMENTATION
cnn.add(tf.keras.layers.RandomFlip('horizontal_and_vertical'))
cnn.add(tf.keras.layers.RandomRotation(0.2))

# CONVOLUTION
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64,64,3)))

# POOLING
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# ADD SECOND & THIRD CONVOLUTION LAYERS
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# FLATTENING
cnn.add(tf.keras.layers.Flatten())

# FULL CONNECTION WITH ANN
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# ADD OUTPUT LAYER
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# =================
# TRAIN & TEST CNN
# =================
# COMPILE
cnn.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# TRAIN & TEST
cnn.fit(x=train_ds, validation_data=val_ds, epochs=100)

# ==================
# SINGLE PREDICTIONS
# ==================
path = 'G:/Documents/Deep Learning/CNN/dataset/single_prediction'
expected_results = ['dog','cat','cat','cat','dog','cat','dog']
print(expected_results)
for file in os.listdir(path):
    test_img = tf.keras.utils.load_img(f'dataset/single_prediction/{file}', target_size=(64,64))
    test_img = tf.keras.utils.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    result = cnn.predict(test_img)
    anml = 'Dog' if result > .5 else 'Cat'
    print(file)
    print(anml)
