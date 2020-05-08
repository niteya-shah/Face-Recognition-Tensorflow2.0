import tensorflow as tf

# We define our loss function, triplet loss which calculates the loss between
# the encodings of a correct prediction and an incorrect prediction.


class TripletLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred)
        anchor, pos, neg = tf.split(y_pred, num_or_size_splits=3, axis=1)
        pos_dist = tf.reduce_sum(input_tensor=tf.square(
            tf.subtract(anchor, pos)), axis=-1)
        neg_dist = tf.reduce_sum(input_tensor=tf.square(
            tf.subtract(anchor, neg)), axis=-1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.margin)
        loss = tf.reduce_sum(input_tensor=tf.maximum(basic_loss, 0.0))
        return loss

    def __init__(self, margin=0.9):
        super().__init__()
        self.margin = margin


class MapLockLoss(tf.keras.losses.Loss):
    # For our custom loss we expect a 2D vector of equal dimensions, to keep
    # complexity in check, the preferable shape is 16X16

    def dist(self, x_pad, y_pad, x, y):
        centric = (x_pad - y_pad) * 0.5 + x - y
        return centric

    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred)
        anchor, pos, neg = tf.split(y_pred, num_or_size_splits=3, axis=1)

        anchor = tf.square(anchor)
        pos = tf.square(pos)
        neg = tf.square(neg)

        anchor_pad = tf.nn.conv2d(tf.expand_dims(tf.squeeze(
            anchor), axis=3), self.ker, strides=[1, 1, 1, 1], padding='SAME')

        pos_pad = tf.nn.conv2d(tf.expand_dims(tf.squeeze(
            pos), axis=3), self.ker, strides=[1, 1, 1, 1], padding='SAME')

        neg_pad = tf.nn.conv2d(tf.expand_dims(tf.squeeze(
            neg), axis=3), self.ker, strides=[1, 1, 1, 1], padding='SAME')

        pos_dist = tf.reduce_sum(input_tensor=self.dist(
            anchor_pad, pos_pad, anchor, pos), axis=-1)
        neg_dist = tf.reduce_sum(input_tensor=self.dist(
            anchor_pad, neg_pad, anchor, neg), axis=-1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.margin)
        return tf.reduce_sum(input_tensor=tf.maximum(basic_loss, 0.0))

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ker = tf.convert_to_tensor(tf.ones((3, 3, 1, 1)))
        self._name_scope = "MapLockLoss"
