import tensorflow as tf
tf.config.optimizer.set_jit(True)
import cv2
import numpy as np
from learning_strategy import online_model  # NOQA
from pathlib import Path

## %%
class NN_fl(object):
    def __init__(self, base_model_weights_path, shape=96):
        path = "/home/touchdown/vggface2_train_preprocessed/"
        self.path = Path(path)
        self.file_dir = [x for x in self.path.iterdir() if x.is_dir()]
        self.shape = shape
        self.base_model_weights_path = base_model_weights_path

    def _get_model(self, base_model_weights_path, train_path):
        base_model = online_model(shape=self.shape, use_trained="Mobile")
        base_model.build(input_shape=[None, self.shape, self.shape, 3])
        base_model.load_weights(self.base_model_weights_path)


        positive_dataset = tf.keras.layers.Input(name='positive_image',
                                                 shape=[self.shape, self.shape, 3],
                                                 dtype=tf.float32)
        negative_dataset = tf.keras.layers.Input(name='negative_image',
                                                 shape=[self.shape, self.shape, 3],
                                                 dtype=tf.float32)

        pos_out = base_model(positive_dataset)
        neg_out = base_model(negative_dataset)
        out = (pos_out - neg_out) * (pos_out - neg_out)

        flat = tf.keras.layers.Flatten()(out)
        layer_1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flat)
        layer_2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(layer_1)
        layer_3 = tf.keras.layers.Dense(32, activation=tf.nn.relu)(layer_2)
        final_layer = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(layer_3)

        fast_learner_model = tf.keras.Model(inputs=[positive_dataset, negative_dataset],
                                                 outputs=final_layer)
        fast_learner_model.compile(optimizer='adam',
                                        loss=tf.losses.CategoricalCrossentropy(),
                                        metrics=['accuracy'])

        try:
            fast_learner_model.load_weights(train_path)
            print("Loaded Weights")
        except Exception:
            pass
        return fast_learner_model

    def _get_data(self, num_people = 1200):
        y_true = list()
        x = np.empty(shape=[num_people, 2, 96, 96, 3])
        for person in range(num_people):
            pos = np.random.randint(len(self.file_dir))
            pos_path = list(self.file_dir[pos].glob("*.jpg"))
            pos_person = np.random.randint(len(pos_path))
            x[person][0] = cv2.cvtColor(cv2.imread(
                pos_path[pos_person].as_posix()), cv2.COLOR_RGB2BGR)
            if(np.random.rand() > 0.5):
                neg_person = np.random.randint(len(pos_path))
                x[person][1] = cv2.cvtColor(cv2.imread(
                    pos_path[neg_person].as_posix()), cv2.COLOR_RGB2BGR)
                y_true.append([False,True])
            else:
                neg = np.random.randint(len(self.file_dir))
                neg_path = list(self.file_dir[neg].glob("*.jpg"))
                neg_person = np.random.randint(len(neg_path))
                x[person][1] = cv2.cvtColor(cv2.imread(
                    neg_path[neg_person].as_posix()), cv2.COLOR_RGB2BGR)
                y_true.append([True, False])
        return x, np.array(y_true)

    def train(self, train_path, num_images=1200 ,iterations=10, validation=False):

        self.train_path = train_path
        if not hasattr(self, "fast_learner_model"):
            self.fast_learner_model = self._get_model()
        x, y = self._get_data(num_images)
        x_1, x_2 = np.split(x,indices_or_sections=2, axis = 1)
        x_1_train = tf.data.Dataset.from_tensors(np.squeeze(x_1))
        x_2_train = tf.data.Dataset.from_tensors(np.squeeze(x_2))
        x_1_f = tf.data.Dataset.zip((x_1_train, x_2_train))
        y_train = tf.data.Dataset.from_tensors(y)
        training_set = tf.data.Dataset.zip(((x_1_f, y_train)))

        if validation:
            x_t, y_t = self._get_data(20)
            x_1_t, x_2_t = np.split(x_t,indices_or_sections=2, axis = 1)
            x_1_test = tf.data.Dataset.from_tensors(np.squeeze(x_1_t))
            x_2_test = tf.data.Dataset.from_tensors(np.squeeze(x_2_t))
            x_1_ft = tf.data.Dataset.zip((x_1_test, x_2_test))
            y_test = tf.data.Dataset.from_tensors(y_t)
            testing_set = tf.data.Dataset.zip(((x_1_ft, y_test)))

        self.fast_learner_model.fit(x=training_set, epochs=iterations,
                                    validation_data=testing_set if validation else None)

        self.fast_learner_model.save_weights(train_path)
