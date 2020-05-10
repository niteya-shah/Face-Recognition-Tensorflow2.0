import os  # NOQA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # NOQA
import tensorflow as tf  # NOQA
tf.config.optimizer.set_jit(True)  # NOQA
import cv2  # NOQA
import numpy as np  # NOQA
from learning_strategy import online_model  # NOQA
from pathlib import Path  # NOQA


class NN_fl(object):
    def __init__(self, train_path, shape=96):
        self.train_path = train_path
        self.shape = shape
        self.base_model = online_model(shape=self.shape, use_trained="Mobile")

    def _get_model(self):

        positive_dataset = tf.keras.layers.Input(name='positive_image',
                                                 shape=[self.shape,
                                                        self.shape, 3],
                                                 dtype=tf.float32)
        negative_dataset = tf.keras.layers.Input(name='negative_image',
                                                 shape=[self.shape,
                                                        self.shape, 3],
                                                 dtype=tf.float32)

        pos_out = self.base_model(positive_dataset)
        neg_out = self.base_model(negative_dataset)
        out = (pos_out - neg_out) * (pos_out - neg_out)

        flat = tf.keras.layers.Flatten()(out)
        layer_1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flat)
        layer_2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(layer_1)
        layer_3 = tf.keras.layers.Dense(32, activation=tf.nn.relu)(layer_2)
        final_layer = tf.keras.layers.Dense(
            2, activation=tf.nn.softmax)(layer_3)

        fast_learner_model = tf.keras.Model(inputs=[positive_dataset,
                                                    negative_dataset],
                                            outputs=final_layer)
        fast_learner_model.compile(optimizer='adam',
                                   loss=tf.losses.CategoricalCrossentropy(),
                                   metrics=['accuracy'])

        try:
            fast_learner_model.load_weights(self.train_path)
            print("Loaded Weights")
        except Exception:
            pass
        return fast_learner_model


class NN_fl_train(NN_fl):

    def __init__(self, img_path, base_model_weights_path, train_path,
                 shape=96):
        super().__init__(train_path, shape)
        self.path = Path(img_path)
        self.file_dir = [x for x in self.path.iterdir() if x.is_dir()]
        self.base_model_weights_path = base_model_weights_path
        self.base_model.build(input_shape=[None, self.shape, self.shape, 3])
        try:
            self.base_model.load_weights(self.base_model_weights_path)
        except Exception:
            pass
        self.fast_learner_model = self._get_model()

    def _get_data(self, num_people=1200):
        y_true = list()
        x = np.empty(shape=[num_people, 2, 96, 96, 3])
        for person in range(num_people):
            pos = np.random.randint(len(self.file_dir))
            pos_path = list(self.file_dir[pos].glob("*.jpg"))
            pos_person = np.random.randint(len(pos_path))
            x[person][0] = cv2.cvtColor(cv2.imread(
                pos_path[pos_person].as_posix()), cv2.COLOR_RGB2BGR)/255
            if(np.random.rand() > 0.5):
                neg_person = np.random.randint(len(pos_path))
                x[person][1] = cv2.cvtColor(cv2.imread(
                    pos_path[neg_person].as_posix()), cv2.COLOR_RGB2BGR)/255
                y_true.append([False, True])
            else:
                neg = np.random.randint(len(self.file_dir))
                neg_path = list(self.file_dir[neg].glob("*.jpg"))
                neg_person = np.random.randint(len(neg_path))
                x[person][1] = cv2.cvtColor(cv2.imread(
                    neg_path[neg_person].as_posix()), cv2.COLOR_RGB2BGR)/255
                y_true.append([True, False])
        return x, np.array(y_true)

    def train(self, num_images=1200, iterations=10, validation=False):

        x, y = self._get_data(num_images)
        x_1, x_2 = np.split(x, indices_or_sections=2, axis=1)
        x_1_train = tf.data.Dataset.from_tensors(np.squeeze(x_1))
        x_2_train = tf.data.Dataset.from_tensors(np.squeeze(x_2))
        x_1_f = tf.data.Dataset.zip((x_1_train, x_2_train))
        y_train = tf.data.Dataset.from_tensors(y)
        training_set = tf.data.Dataset.zip(((x_1_f, y_train)))
        training_set.prefetch(3)
        if validation:
            x_t, y_t = self._get_data(20)
            x_1_t, x_2_t = np.split(x_t, indices_or_sections=2, axis=1)
            x_1_test = tf.data.Dataset.from_tensors(np.squeeze(x_1_t))
            x_2_test = tf.data.Dataset.from_tensors(np.squeeze(x_2_t))
            x_1_ft = tf.data.Dataset.zip((x_1_test, x_2_test))
            y_test = tf.data.Dataset.from_tensors(y_t)
            testing_set = tf.data.Dataset.zip(((x_1_ft, y_test)))
            testing_set.prefetch(3)

        self.fast_learner_model.fit(x=training_set, epochs=iterations,
                                    validation_data=testing_set if validation
                                    else None)

        self.fast_learner_model.save_weights(self.train_path)


if __name__ == "__main__":

    train = NN_fl_train("/home/touchdown/vggface2_train_preprocessed",
                        "./weights/siamese_weights_3.h5",
                        "./weights/fast_learner_weights_1.h5")
    for i in range(50):
        train.train(100, 4, True)
