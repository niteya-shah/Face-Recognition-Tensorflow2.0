import tensorflow as tf
from model_arch import BaseModel
import tensorflow_addons as tfa
from losses import TripletLoss


def offline_model(shape=96, loss=TripletLoss(),
                  optimizer=tf.keras.optimizers.Adam(0.0001)):
    # Create a Tensorflow model, which uses an offline model.
    truth_dataset = tf.keras.layers.Input(name='truth_dataset',
                                          shape=[shape, shape, 3],
                                          dtype=tf.float32)
    positive_dataset = tf.keras.layers.Input(name='positive_image',
                                             shape=[shape, shape, 3],
                                             dtype=tf.float32)
    negative_dataset = tf.keras.layers.Input(name='negative_image',
                                             shape=[shape, shape, 3],
                                             dtype=tf.float32)

    model = BaseModel()
    truth_val = tf.reshape(model.call(truth_dataset), [-1, 16, 16])
    positive_val = tf.reshape(model.call(positive_dataset), [-1, 16, 16])
    negative_val = tf.reshape(model.call(negative_dataset), [-1, 16, 16])

    concat_layer = tf.stack([truth_val, positive_val, negative_val], axis=1)
    siamese_model = tf.keras.Model(
        inputs=[truth_dataset, positive_dataset, negative_dataset],
        outputs=concat_layer)

    siamese_model.compile(optimizer=optimizer, loss=loss)

    return siamese_model


def online_model(shape=96, use_trained="Mobile",
                 optimizer=tf.keras.optimizers.Adam(0.0001),
                 loss=tfa.losses.TripletSemiHardLoss(margin=1.0)):
    # Generate an online learning model.
    images = tf.keras.layers.Input(
        name="Input", shape=[shape, shape, 3], dtype=tf.float32)

    if use_trained != "None":

        if use_trained == "Mobile":
            model = tf.keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(shape, shape, 3), include_top=False,
                weights="imagenet")
            print("using Mobilenet")

        elif use_trained == "Inception":
            model = tf.keras.applications.InceptionV3(
                input_shape=(shape, shape, 3), include_top=False)
            print("using Inception")

        features = model(images)
        pooled_features = tf.keras.layers.GlobalAveragePooling2D()(features)
        dense_features = tf.keras.layers.Dense(
            128, activation=None)(pooled_features)
        embeddings = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(
            tf.cast(x, dtype='float32'), axis=1))(dense_features)
    else:

        print("using base model")
        model = BaseModel()
        embeddings = model.call(images)

    siamese_model = tf.keras.Model(inputs=images, outputs=embeddings)
    siamese_model.compile(optimizer=optimizer, loss=loss)

    return siamese_model
