import numpy as np
import cv2
from write_dataset import process_img
from pathlib import Path
from fast_learner import NN_fl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # NOQA


file_path = Path("./images")

NN_fl_path = "./weights/fast_learner_weights_1.h5"
w = NN_fl("./weights/siamese_weights_3.h5")
model = w._get_model("./weights/siamese_weights_3.h5", NN_fl_path)

cap = cv2.VideoCapture(0)
img_gen = process_img(96)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    found_img = img_gen.pre_process(frame)
    if found_img is not None:
        images = file_path.glob("*.jpg")
        for image in images:
            img_p = cv2.imread(image.as_posix())
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
            cv2.imwrite("./images/"+name+".jpg", img)
        else:
            print("no face found")

    if k % 256 == 32:
        cap.release()
        cv2.destroyAllWindows()
        break
