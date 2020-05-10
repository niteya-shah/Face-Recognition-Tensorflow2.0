import os  # NOQA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # NOQA
import numpy as np  # NOQA
import cv2  # NOQA
from write_dataset import process_img  # NOQA
from pathlib import Path  # NOQA
from fast_learner import NN_fl  # NOQA


file_path = Path("./images")

NN_fl_path = "./weights/fast_learner_weights_1.h5"
w = NN_fl(NN_fl_path)
model = w._get_model()

cap = cv2.VideoCapture(0)
img_gen = process_img(96)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    found_img = img_gen.pre_process(frame)
    if found_img is not None:
        images = file_path.glob("*.jpg")
        for image in images:
            img_p = cv2.cvtColor(
                cv2.imread(image.as_posix()), cv2.COLOR_RGB2BGR)/255
            prediction = model.predict([img_p[np.newaxis],
                                        found_img[np.newaxis]])
            if(prediction.T[1] > 0.8):
                print("Found " + image.as_posix())
                break

    cv2.imshow("image", frame)

    k = cv2.waitKey(1)
    if k % 256 == 13:
        name = input("Please input name: ")
        img = img_gen.pre_process(frame)
        if img is not None:
            cv2.imwrite("./images/"+name+".jpg", img * 255)
        else:
            print("no face found")

    if k % 256 == 32:
        cap.release()
        cv2.destroyAllWindows()
        break
