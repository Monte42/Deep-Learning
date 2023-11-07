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



# CONVOLUTION



# POOLING



# ADD SECOND CONVOLUTION



# FLATTENING



# FULL CONNECTION WITH ANN



# ADD OUTPUT LAYER





# ==========
# TRAIN CNN
# ==========



# =========
# TEST CNN
# =========