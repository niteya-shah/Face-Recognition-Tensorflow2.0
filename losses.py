import tensorflow as tf

#We define our loss function, triplet loss which calculates the loss between the encodings of a correct prediction and an incorrect prediction.
class TripletLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred)
        anchor, pos, neg = tf.split(y_pred,num_or_size_splits = 3, axis = 1)
        pos_dist   = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, pos)), axis=-1)
        neg_dist   = tf.reduce_sum(input_tensor=tf.square(tf.subtract(anchor, neg)), axis=-1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.margin)
        loss       = tf.reduce_sum(input_tensor=tf.maximum(basic_loss, 0.0))
        return loss

    def __init__(self, margin = 0.9):
        super().__init__()
        self.margin = margin
