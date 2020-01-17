import dlib
from skimage import io
ff = dlib.get_frontal_face_detector()
img = io.imread("lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
img = np.flip(img, axis = [0,1])
boundary = ff(img, 1)[0]
boundary
io.imshow(img[boundary.top():boundary.bottom(),boundary.left():boundary.right()])

import numpy as np
