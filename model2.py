# We import all the necessary libraries and define parameters for our model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # NOQA

import tensorflow as tf
tf.config.optimizer.set_jit(True)
from tensorflow.keras.mixed_precision import experimental as mixed_precision  # NOQA
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
import numpy as np  # NOQA
from fetch_data import image_generator_online, image_generator_offline  # NOQA
from learning_strategy import online_model, offline_model  # NOQA
from tensorflow_addons.losses import TripletHardLoss, TripletSemiHardLoss   # NOQA
from losses import MapLockLoss, TripletLoss  # NOQA

shape = 96
path = "/home/touchdown/vggface2_train_preprocessed/"
log_dir = "/D/work/ML/Faces/src/logs/"

# We set GPU parameters to ensure our GPU doesnt run out of memory and uses it
# in an efficient manner.
# tf.config.experimental.set_memory_growth = True
# tf.config.experimental.log_device_placement = True
AUTOTUNE = tf.data.experimental.AUTOTUNE
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

method = "online"

# The model is compiled and then fit on the dataset
if method == "online":
    siamese_model = online_model(shape=96, use_trained="Mobile",
                                 optimizer=tf.keras.optimizers.Adam(0.01),
                                 loss=TripletSemiHardLoss(margin=0.2))
    dataset = image_generator_online(path, K=10, num_people=15).return_val()
    dataset = dataset.repeat(3).prefetch(AUTOTUNE)
else:
    siamese_model = offline_model(shape=96, optimizer=tf.keras.optimizers.Adam(
        0.001),                             loss=MapLockLoss(margin=0.3))
    dataset = image_generator_offline(path, K=512).return_val()
    dataset = dataset.prefetch(AUTOTUNE)

try:
    siamese_model.load_weights("/D/work/ML/Faces/src/weights/siamese_weights_5.h5")
except Exception:
    pass
# We define our model which a Combination of Convulational layers followed by
# an encoding dense layer and then Calulate the loss of the model.
siamese_model.fit(dataset, epochs=150, use_multiprocessing=True,
                  callbacks=[tensorboard_callback])
# siamese_model.save_weights("/D/work/ML/Faces/src/weights/siamese_weights_6.h5")
