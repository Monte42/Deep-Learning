# IMPORTS
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# =====================
# DATA PRE_PROCESSING
# =====================
# TRAIN SET
train_data_gen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = .2,
    zoom_range = .2,
    horizontal_flip = True,
)

train_set = train_data_gen.flow_from_directory(
    'dataset/training_set',
    target_size = (64,64),
    batch_size = 32,
    class_mode = 'binary'
)

# TEST SET
test_data_gen = ImageDataGenerator(rescale=1./255)

test_set = test_data_gen(
    'dataset/test_set',
    target_size = (64,64),
    batch_size = 32,
    class_mode = 'binary'
)


# =================================
# BUILD CONVOLUTION NEURAL NETWORK
# =================================
# INITIALIZE CONVOLUTION NEURAL NETWORK
cnn = tf.keras.models.Sequential()


# CONVOLUTION
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64,64,3)))

# POOLING
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, stides=2))

# ADD SECOND CONVOLUTION
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, stides=2))

# FLATTENING
cnn.add(tf.keras.layers.Flatten())

# FULL CONNECTION WITH ANN
ann = tf.keras.models.Sequential()
ann.add(tk.keras.layers.Dense(units=128, activation='relu'))
ann.add(tk.keras.layers.Dense(units=128, activation='relu'))
ann.add(tk.keras.layers.Dense(units=128, activation='relu'))

# ADD OUTPUT LAYER
ann.add(tk.keras.layers.Dense(units=1, activation='sigmoid'))

# ==========
# TRAIN CNN
# ==========



# =========
# TEST CNN
# =========