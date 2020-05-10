import cv2
import socket
import numpy as np
import pickle
from write_dataset import process_img
from utils import take_photo, show_photo
from fast_learner import NN_fl

proc = process_img(96)

img = take_photo()
show_photo(img)
img_p = proc.pre_process(img)
show_photo(img_p)


img_1 = take_photo()
img_p_1 = proc.pre_process(img_1)
show_photo(img_p_1)

NN_fl_path = "./weights/fast_learner_weights_1.h5"
w = NN_fl("./weights/siamese_weights_3.h5")
model = w._get_model("./weights/siamese_weights_3.h5", NN_fl_path)

prediction = model.predict([img_p[np.newaxis], img_p_1[np.newaxis]])
result = prediction.T[1] > 0.8
print(result)
