import warnings
warnings.simplefilter('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
tf.autograph.set_verbosity(0)
import matplotlib.pyplot as plt
from learning_strategy import online_model
import imageio
from skimage.transform import resize
import numpy as np
from pathlib import Path
import cv2
from sklearn.neighbors import RadiusNeighborsClassifier
import sklearn.metrics
from scipy import interp
import matplotlib.pyplot as plt

model = online_model(shape = 96, use_trained = "Mobile")
model.build(input_shape = [None,96,96,3])
status = model.load_weights("./weights/siamese_weights_3.h5")
## %%

path2 = "/home/touchdown/vggface2_train_preprocessed/"
path = Path(path2)

import gc
gc.collect()
file_dir = [x for x in path.iterdir() if x.is_dir()]

num_people = 3000
people = list()
num_img = list()

for person in range(num_people):
    for image in list(file_dir[person].glob('*.jpg'))[0:50]:
        people.append(cv2.cvtColor(cv2.imread(image.as_posix()), cv2.COLOR_RGB2BGR))
        num_img.append(person)

people = np.array(people)
gc.collect()
num_img = np.array(num_img)
embeddings = model.predict(people)
del people
gc.collect()
neigh = RadiusNeighborsClassifier(radius = 0.025, n_jobs = 7)
neigh.fit(embeddings, num_img)
t = neigh.predict(embeddings)
print("Accuracy is {}".format(sklearn.metrics.accuracy_score(t, num_img) * 100))
print("Precision is {}".format(sklearn.metrics.precision_score(t, num_img, average='weighted')))
print("Recall is {}".format(sklearn.metrics.recall_score(t, num_img, average='macro')))
