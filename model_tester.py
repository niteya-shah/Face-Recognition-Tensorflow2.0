import numpy as np  # NOQA
from write_dataset import process_img  # NOQA
from utils import take_photo, show_photo  # NOQA
from fast_learner import NN_fl  # NOQA

proc = process_img(96)

img = take_photo()
show_photo(img)
img_p = proc.pre_process(img)
show_photo(img_p)


img_1 = take_photo()
img_p_1 = proc.pre_process(img_1)
show_photo(img_p_1)

NN_fl_path = "./weights/fast_learner_weights_1.h5"
w = NN_fl(NN_fl_path)
model = w._get_model()

prediction = model.predict([img_p[np.newaxis], img_p_1[np.newaxis]])
result = prediction.T[1] > 0.8
print(result)
