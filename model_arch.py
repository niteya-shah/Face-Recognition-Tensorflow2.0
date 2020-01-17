import tensorflow as tf

def convblock(model, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1
    filters = int(filters * alpha)
    model.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv_sw_pad'))
    model.add(tf.keras.layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1'))
    model.add(tf.keras.layers.BatchNormalization(axis=channel_axis, name='conv_sw_bn'))
    model.add(tf.keras.layers.ReLU(6., name='conv1_relu'))


def depthwiseconvblock(model, pointwiseconvfilters, alpha, depth_multiplier=1, strides=(1, 1), block_name=1):
    channel_axis = -1
    pointwiseconvfilters = int(pointwiseconvfilters * alpha)
    if strides != (1, 1):
        model.add(tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_dw_pad_%d' % block_name))
    model.add(tf.keras.layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_name))
    model.add(tf.keras.layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_name))
    model.add(tf.keras.layers.ReLU(6., name='conv_dw_%d_relu' % block_name))

    model.add(tf.keras.layers.Conv2D(pointwiseconvfilters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_name))
    model.add(tf.keras.layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_name))
    model.add(tf.keras.layers.ReLU(6., name='conv_pw_%d_relu' % block_name))


def model_arch(alpha = 0.25, depth_multiplier = 1, dropout = 1e-3, values = 128):
    model = tf.keras.Sequential()
    convblock(model, 32, alpha, strides=(2, 2))
    depthwiseconvblock(model, 64, alpha, depth_multiplier, block_name=1)

    depthwiseconvblock(model, 128, alpha, depth_multiplier,strides=(2, 2), block_name=2)
    depthwiseconvblock(model, 128, alpha, depth_multiplier, block_name=3)

    depthwiseconvblock(model, 256, alpha, depth_multiplier, strides=(2, 2), block_name=4)
    depthwiseconvblock(model, 256, alpha, depth_multiplier, block_name=5)

    depthwiseconvblock(model, 512, alpha, depth_multiplier,strides=(2, 2), block_name=6)
    depthwiseconvblock(model, 512, alpha, depth_multiplier, block_name=7)
    depthwiseconvblock(model, 512, alpha, depth_multiplier, block_name=8)
    depthwiseconvblock(model, 512, alpha, depth_multiplier, block_name=9)
    depthwiseconvblock(model, 512, alpha, depth_multiplier, block_name=10)
    depthwiseconvblock(model, 512, alpha, depth_multiplier, block_name=11)

    depthwiseconvblock(model, 1024, alpha, depth_multiplier, strides=(2, 2), block_name=12)
    depthwiseconvblock(model, 1024, alpha, depth_multiplier, block_name=13)
    shape = (1, 1, int(1024 * alpha))

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Reshape(shape, name='reshape_1'))
    model.add(tf.keras.layers.Dropout(dropout, name='dropout'))
    model.add(tf.keras.layers.Conv2D(values, (1, 1), padding='same', name='conv_preds'))
    model.add(tf.keras.layers.Reshape((values,), name='reshape_2'))
    model.add(tf.keras.layers.Activation('softmax', name='act_softmax'))

    return model
