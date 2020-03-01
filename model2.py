#We import all the necessary libraries and define parameters for our model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.config.optimizer.set_jit(True)
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
import numpy as np
from fetch_data import image_generator_online
from learning_strategy import online_model
import tensorflow_addons as tfa

shape = 96
path = "/home/touchdown/vggface2_train_preprocessed/"

#We set GPU parameters to ensure our GPU doesnt run out of memory and uses it in an efficient manner.
# tf.config.experimental.set_memory_growth = True
# tf.config.experimental.log_device_placement = True
AUTOTUNE = tf.data.experimental.AUTOTUNE

#The model is compiled and then fit on the dataset
siamese_model = online_model(shape = 96, use_trained = "Mobile",optimizer = tf.keras.optimizers.Adam(0.01), loss = tfa.losses.TripletSemiHardLoss(margin = 0.2))
dataset = image_generator_online(path, K = 10, num_people = 15).return_val()
dataset = dataset.repeat(3).prefetch(AUTOTUNE)

siamese_model.load_weights("/D/work/ML/Faces/src/weights/siamese_weights_3.h5")
#We define our model which a Combination of Convulational layers followed by an encoding dense layer and then Calulate the loss of the model.
siamese_model.fit(dataset,epochs = 15000, use_multiprocessing = True)

siamese_model.save_weights("/D/work/ML/Faces/src/weights/siamese_weights_4.h5")
