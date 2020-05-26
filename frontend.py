import os  # NOQA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # NOQA
import numpy as np  # NOQA
import cv2  # NOQA
from write_dataset import process_img  # NOQA
from pathlib import Path  # NOQA
from fast_learner import NN_fl  # NOQA
import time


class Gen_frame(object):
    def __init__(self,  mydb, frame_time=1):
        self.file_path = Path("./static/images")

        NN_fl_path = "./weights/fast_learner_weights_1.h5"
        w = NN_fl(NN_fl_path)
        self.model = w._get_model()
        self.img_gen = process_img(96)
        self.start_time = time.time()
        self.frame_time = frame_time

    def box_image(self, image, boundary, name):
        image = cv2.rectangle(image, (boundary.tl_corner().x,
                                      boundary.tl_corner().y),
                                     (boundary.br_corner().x,
                                      boundary.br_corner().y),
                                     (255, 0, 0), thickness=2)
        return cv2.putText(frame,
                           image.as_posix().replace(".jpg", "")
                           .replace("static/images/", ""),
                           (boundary.br_corner().x - 60,
                            boundary.br_corner().y - 5), 3, 0.4, 255)

    def find_name(self, prediciton, name_store):
        return name_store[numpy.argmax(prediction.T[1])]

    def predict_image(self, image):
        found_img = self.img_gen.pre_process(image, True)
        if found_img is not None:
            found_img, boundary = found_img
            images = self.file_path.glob("*.jpg")
            img_store = list()
            name_store = list()
            for image in images:
                img_store.append(cv2.cvtColor(cv2.imread(image.as_posix()),
                                              cv2.COLOR_RGB2BGR)/255)
                name_store.append(image.as_posix())
            img_store, found_img = np.broadcast_arrays(img_store, found_img)
            prediction = self.model.predict([img_store, found_img])

            return prediction, name_store, boundary
        return None

    def gen_video(self, frame):
        if ((frame is not None):
            result=predict_image(frame)
            if result is None:
                name=self.find_name(result[0], result[1])
                frame=self.box_image(frame, result[2], name)
                data=cv2.imencode('.png', frame)[1].tobytes()
                return (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n')
            else:
                return (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    def gen_image(self):
        while(True):
            # Capture frame-by-frame
            _, frame=self.cap.read()
            if ((frame is not None) and
               (int(time.time() - self.start_time) % self.frame_time == 0)):
                self.found_img=self.img_gen.pre_process(frame, True)[0]

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.png', frame)[1].tobytes() + b'\r\n\r\n')

    def show_video(self):
        while(True):
            # Capture frame-by-frame
            _, frame=self.cap.read()
            if frame is not None and int(time.time() - self.start_time) % self.frame_time == 0:
                found_img=self.img_gen.pre_process(frame, True)
                if found_img is not None:
                    found_img, boundary=found_img
                    frame=cv2.rectangle(frame, (boundary[0].tl_corner().x,
                                                  boundary[0].tl_corner().y),
                                          (boundary[0].br_corner().x,
                                           boundary[0].br_corner().y),
                                          (255, 0, 0), thickness=2)

                    print("\033[2K", end="\r")
                    images=self.file_path.glob("*.jpg")
                    for image in images:
                        img_p=cv2.cvtColor(
                            cv2.imread(image.as_posix()), cv2.COLOR_RGB2BGR)/255
                        prediction=self.model.predict([img_p[np.newaxis],
                                                         found_img[np.newaxis]])
                        if(prediction.T[1] > 0.8):
                            cv2.putText(frame,
                                        image.as_posix().replace(".jpg",
                                                                 "").replace("images/",
                                                                             ""),
                                        (boundary[0].br_corner().x - 60,
                                         boundary[0].br_corner().y - 5), 3, 0.4, 255)

            cv2.imshow("VideoCapture", frame)

            k=cv2.waitKey(1)
            if k % 256 == 13:
                name=input("Please input name: ")
                img=self.img_gen.pre_process(frame)
                if img is not None:
                    cv2.imwrite("./static/images/"+name+".jpg", img * 255)
                else:
                    print("no face found")

            if k % 256 == 32:
                self.cap.release()
                cv2.destroyAllWindows()
                break

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
