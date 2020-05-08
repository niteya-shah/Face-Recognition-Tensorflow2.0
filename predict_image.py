import warnings  # NOQA
warnings.simplefilter('ignore')  # NOQA
import os  # NOQA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # NOQA
os.environ['AUTOGRAPH_VERBOSITY'] = '0'  # NOQA
import tensorflow as tf  # NOQA
tf.autograph.set_verbosity(0)  # NOQA
import matplotlib.pyplot as plt  # NOQA
from learning_strategy import online_model  # NOQA
import imageio  # NOQA
from skimage.transform import resize  # NOQA
import numpy as np  # NOQA
from pathlib import Path  # NOQA
import cv2  # NOQA
from sklearn.neighbors import RadiusNeighborsClassifier  # NOQA
import sklearn.metrics  # NOQA
import matplotlib.pyplot as plt  # NOQA
import gc  # NOQA

model = online_model(shape=96, use_trained="Mobile")
model.build(input_shape=[None, 96, 96, 3])
status = model.load_weights("./weights/siamese_weights_3.h5")

path2 = "/home/touchdown/vggface2_train_preprocessed/"
path = Path(path2)


gc.collect()
file_dir = [x for x in path.iterdir() if x.is_dir()]

num_people = 3000
people = list()
num_img = list()

for person in range(num_people):
    for image in list(file_dir[person].glob('*.jpg'))[0:50]:
        people.append(cv2.cvtColor(cv2.imread(
            image.as_posix()), cv2.COLOR_RGB2BGR))
        num_img.append(person)

people = np.array(people)
gc.collect()
num_img = np.array(num_img)
embeddings = model.predict(people)
del people
gc.collect()
neigh = RadiusNeighborsClassifier(radius=0.018, n_jobs=7)
neigh.fit(embeddings, num_img)
t = neigh.predict(embeddings)
print("Accuracy is {}".format(sklearn.metrics.accuracy_score(t,
                                                             num_img) * 100))
print("Precision is {}".format(
    sklearn.metrics.precision_score(t, num_img, average='weighted')))
print("Recall is {}".format(
    sklearn.metrics.recall_score(t, num_img, average='macro')))
