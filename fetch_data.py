from pathlib import Path
import cv2
import random
import numpy as np
import tensorflow as tf


class image_generator_offline:
    # Generate 3 images per training shot.
    def gen_image(self):
        positive = random.randint(0, len(self.file_dir) - 1)
        negative = random.randint(0, len(self.file_dir) - 1)
        pos_path = list(self.file_dir[positive].glob('*.jpg'))
        neg_path = list(self.file_dir[negative].glob('*.jpg'))
        image_list = [cv2.cvtColor(cv2.imread(images.as_posix()),
                                   cv2.COLOR_RGB2BGR)
                      for images in random.choices(pos_path, k=2)]
        image_list.append(cv2.cvtColor(cv2.imread(random.choice(
            neg_path).as_posix()), cv2.COLOR_RGB2BGR))
        return np.array(image_list, dtype=np.float32)

    def _generator(self):
        while(1):
            x, y, z = [np.squeeze(i)
                       for i in np.split(self.gen_image(), 3, axis=0)]
            yield x, y, z

    def __init__(self, path, K=4, shape=96):
        self.path = Path(path)
        self.shape = shape
        self.file_dir = [x for x in self.path.iterdir() if x.is_dir()]
        self.K = K

    def return_val(self):
        shape_tuple = ((self.shape, self.shape, 3),) * 3
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=shape_tuple).batch(self.K)
        return tf.data.Dataset.zip((dataset,
                                    tf.data.Dataset.from_tensors(
                                        tf.zeros([1]))))


class image_generator_online:
    # Generate images according online learning strategy.
    def gen_image(self):
        image_list = np.zeros(
            [self.num_people * self.K, self.shape, self.shape, 3])
        label_list = np.zeros([self.num_people * self.K])
        for person in range(self.num_people):
            counter = 0
            positive = random.randint(0, len(self.file_dir) - 1)
            pos_path = list(self.file_dir[positive].glob('*.jpg'))
            for image in random.choices(pos_path, k=self.K):
                image_list[person * self.K + counter] = cv2.cvtColor(
                    cv2.imread(image.as_posix()), cv2.COLOR_RGB2BGR)/255
                label_list[person * self.K + counter] = person
                counter += 1
        yield image_list, label_list

    def __init__(self, path, K=4, num_people=32, shape=96):
        self.path = Path(path)
        self.shape = shape
        self.file_dir = [x for x in self.path.iterdir() if x.is_dir()]
        self.K = K
        self.num_people = num_people

    def return_val(self):
        return tf.data.Dataset.from_generator(
            self.gen_image,
            output_types=(tf.float32, tf.int32),
            output_shapes=((self.K * self.num_people, self.shape,
                            self.shape, 3), (self.K * self.num_people))
        )


if __name__ == "__main__":
    path = "/D/work/ML/Faces/VGGFace2/vggface2_train_preprocessed/"
    gen = image_generator_offline(path, 96)
    t = gen._generator().__next__()
    gen2 = image_generator_online(path, 4, 32)
    t2 = gen2.return_val()
