# import warnings
# warnings.simplefilter('ignore')
import tensorflow as tf
import numpy as np
from fetch_data import image_data
import os
import matplotlib.pyplot as plt
from model_arch import model_arch
import math
import imageio
from skimage.transform import resize


model = model_arch()
model.build(input_shape = [1,224,224,3])
status = model.load_weights("./weights/model_weights_1.h5")
data = image_data(os.getcwd() + '/lfw-deepfunneled/', 100, 224)
t = model.predict(data[0][r])
t1 = model.predict(data[1][r])
t2 = model.predict(data[2][r])
print(np.linalg.norm(t - t1))
print(np.linalg.norm(t - t2))
## %%
r = 29
plt.imshow(np.squeeze(data[0][r]))
plt.imshow(np.squeeze(data[1][r]))
plt.imshow(np.squeeze(data[2][r]))
## %%

im1 = resize(imageio.imread('/D/work/ML/Faces/PINS/pins_barbara palvin face/barbara palvin face32.jpg'), [224,224,3])
im2 = resize(imageio.imread('/D/work/ML/Faces/PINS/pins_Chance Perdomo/Chance Perdomo84.jpg'), [224,224,3])
t1 = model.predict(np.reshape(im1,[1,224,224,3]))
t2 = model.predict(np.reshape(im2,[1,224,224,3]))
print(np.linalg.norm(t1 - t2))
plt.imshow(np.squeeze(im1))
plt.imshow(np.squeeze(im2))
np.where(np.squeeze(t1) > 0.001)
np.where(np.squeeze(t2) > 0.001)
