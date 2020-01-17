#We import all the necessary libraries and define parameters for our model
import tensorflow as tf
import numpy as np
from fetch_data import image_generator
import os
import matplotlib.pyplot as plt
from model_arch import model_arch
import math
import sys
from pathlib import Path
import random
import imageio
import pydot
import graphviz

alpha = 0.5
shape = 96
margin = tf.constant(0.2, dtype = tf.float32)
batch_size = 30
images_per_batch = math.ceil(256/32)

#We define our loss function, triplet loss which calculates the loss between the encodings of a correct prediction and an incorrect prediction.
class TripletLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred)
        anchor, pos, neg = tf.split(y_pred,num_or_size_splits = 3, axis = 1)
        pos_dist   = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, pos)), axis=-1)
        neg_dist   = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, neg)), axis=-1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)
        loss       = tf.reduce_sum(input_tensor=tf.maximum(basic_loss, 0.0))
        return loss

#We set GPU parameters to ensure our GPU doesnt run out of memory and uses it in an efficient manner.
tf.config.experimental.set_memory_growth = True
tf.config.experimental.log_device_placement = True

AUTOTUNE = tf.data.experimental.AUTOTUNE

#We fetch the Image dataset in an numpy array from the image_data function.
truth_dataset = tf.keras.layers.Input(name = 'truth_dataset', shape = [shape,shape,3], dtype = tf.float32)
positive_dataset = tf.keras.layers.Input(name = 'positive_image', shape = [shape,shape,3], dtype = tf.float32)
negative_dataset = tf.keras.layers.Input(name = 'negative_image', shape = [shape,shape,3], dtype = tf.float32)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128))

model = model_arch()

truth_val = model(truth_dataset)
positive_val = model(positive_dataset)
negative_val = model(negative_dataset)

concat_layer = tf.stack([truth_val, positive_val, negative_val], axis = 1)
siamese_model = tf.keras.Model(inputs = [truth_dataset, positive_dataset, negative_dataset], outputs = concat_layer)
tb = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    #The model is compiled and then fit on the dataset
triplet_loss = TripletLoss()
siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),loss = triplet_loss)

# if(int(sys.argv[1]) == 1):
#     path = os.getcwd() + '/lfw-deepfunneled/'
#     print("lfw")
# else:

path = "/D/work/ML/Faces/VGGFace2/vggface2_train_preprocessed"
gen = image_generator(path,1000, 96)
truth_image_1,positive_anchor_1,negative_anchor_1 = gen.image_data()

# for image imshow use astype np.float32
#We organize our data in tensors using Tensorflows Dataset API, after which we batch them, shuffle them and allocate the memory.
truth_image = tf.data.Dataset.from_tensor_slices(np.squeeze(truth_image_1))
positive_anchor = tf.data.Dataset.from_tensor_slices(np.squeeze(positive_anchor_1))
negative_anchor = tf.data.Dataset.from_tensor_slices(np.squeeze(negative_anchor_1))

dataset = tf.data.Dataset.zip((truth_image, positive_anchor, negative_anchor))
dataset = tf.data.Dataset.zip((dataset,tf.data.Dataset.from_tensor_slices(tf.zeros([1000]))))
dataset = dataset.batch(batch_size).shuffle(100).repeat(100).prefetch(buffer_size=AUTOTUNE)
#We define our model which a Combination of Convulational layers followed by an encoding dense layer and then Calulate the loss of the model.
siamese_model.fit(dataset,epochs = 30, steps_per_epoch = 25)

siamese_model.save_weights("./weights/siamese_weights_1.h5")
model.save_weights("./weights/model_weights_1.h5")

tf.keras.utils.plot_model(model, show_shapes = True, expand_nested = True)
