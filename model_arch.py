import tensorflow as tf
import tensorflow_addons as tfa

class BaseModel(tf.keras.Model):

	def __init__(self, values = 128):
		super(BaseModel, self).__init__()
		#conv block
		channel_axis = 1
		filters = 32
		kernel = (3, 3)
		self.zp_conv = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))
		self.conv2d_conv = tf.keras.layers.Conv2D(filters, kernel, padding='valid',use_bias=False,strides=(2,2))
		self.bn_conv = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_conv =  tf.keras.layers.ReLU(6.)

		depth_multiplier = 1
		channel_axis = -1
		#depthwise layer 1
		pointwiseconvfilters = 16
		strides = (1, 1)
		self.dc2d_d1 = tf.keras.layers.DepthwiseConv2D((3, 3),
								padding='same' if strides == (1, 1) else 'valid',
								depth_multiplier=depth_multiplier,
								strides=strides,
								use_bias=False)
		self.bn_d1 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d1  = tf.keras.layers.ReLU(6.)

		#depthwise layer 2
		pointwiseconvfilters = 32
		strides = (2, 2)
		self.zp_d2 = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)))
		self.dc2d_d2 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d2 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d2  = tf.keras.layers.ReLU(6.)

		#depthwise layer 3
		pointwiseconvfilters = 32
		strides = (1, 1)
		self.dc2d_d3 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d3 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d3  = tf.keras.layers.ReLU(6.)

		#depthwise layer 4
		pointwiseconvfilters = 64
		strides = (2, 2)
		self.zp_d4 = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)))
		self.dc2d_d4 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d4 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d4  = tf.keras.layers.ReLU(6.)

		#depthwise layer 5
		pointwiseconvfilters = 64
		strides = (1, 1)
		self.dc2d_d5 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d5 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d5  = tf.keras.layers.ReLU(6.)

		#depthwise layer 6
		pointwiseconvfilters = 128
		strides = (2, 2)
		self.zp_d6 = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)))
		self.dc2d_d6 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d6 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d6  = tf.keras.layers.ReLU(6.)

		#depthwise layer 7
		pointwiseconvfilters = 128
		strides = (1, 1)
		self.dc2d_d7 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d7 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d7  = tf.keras.layers.ReLU(6.)

		#depthwise layer 8
		pointwiseconvfilters = 128
		strides = (1, 1)
		self.dc2d_d8 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d8 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d8  = tf.keras.layers.ReLU(6.)

		#depthwise layer 9
		pointwiseconvfilters = 128
		strides = (1, 1)
		self.dc2d_d9 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d9 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d9  = tf.keras.layers.ReLU(6.)

		#depthwise layer 10
		pointwiseconvfilters = 128
		strides = (1, 1)
		self.dc2d_d10 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d10 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d10  = tf.keras.layers.ReLU(6.)

		#depthwise layer 11
		pointwiseconvfilters = 128
		strides = (1, 1)
		self.dc2d_d11 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d11 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d11  = tf.keras.layers.ReLU(6.)

		#depthwise layer 12
		pointwiseconvfilters = 256
		strides = (2, 2)
		self.zp_d12 = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)))
		self.dc2d_d12 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d12 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d12  = tf.keras.layers.ReLU(6.)

		#depthwise layer 13
		pointwiseconvfilters = 256
		strides = (1, 1)
		self.dc2d_d13 = tf.keras.layers.DepthwiseConv2D((3, 3),
								   padding='same' if strides == (1, 1) else 'valid',
								   depth_multiplier=depth_multiplier,
								   strides=strides,
								   use_bias=False)
		self.bn_d13 = tf.keras.layers.BatchNormalization(axis=channel_axis)
		self.relu_d13  = tf.keras.layers.ReLU(6.)

		#final block
		shape = (1, 1, 72)

		self.glavgpool2d_f = tf.keras.layers.GlobalAveragePooling2D()
		self.res1_f = tf.keras.layers.Reshape(shape)
		self.den_128 = tf.keras.layers.Dense(128)
		self.drp_f = tf.keras.layers.Dropout(1e-3)
		self.conv2d_f = tf.keras.layers.Conv2D(values, (1, 1), padding='same')
		self.res2_f = tf.keras.layers.Reshape((values,))
		self.sft_f = tf.keras.layers.Activation('softmax', dtype='float32')
		self.norm_f = tf.keras.layers.Lambda(lambda x:tf.math.l2_normalize(tf.cast(x,dtype='float32'), axis = 1))
		self.flat_f = tf.keras.layers.Flatten()

	def call(self, input):
		#conv
		model = self.zp_conv(input)
		model = self.conv2d_conv(model)
		model = self.bn_conv(model)
		model = self.relu_conv(model)

		#dp 1
		model = self.dc2d_d1(model)
		model = self.bn_d1(model)
		model = self.relu_d1(model)

		#dp 2
		model = self.zp_d2(model)
		model = self.dc2d_d2(model)
		model = self.bn_d2(model)
		model = self.relu_d2(model)

		#dp 3
		model = self.dc2d_d3(model)
		model = self.bn_d3(model)
		model = self.relu_d3(model)

		#dp 4
		model = self.zp_d4(model)
		model = self.dc2d_d4(model)
		model = self.bn_d4(model)
		model = self.relu_d4(model)

		#dp 5
		model = self.dc2d_d5(model)
		model = self.bn_d5(model)
		model = self.relu_d5(model)

		#dp 6
		model = self.zp_d6(model)
		model = self.dc2d_d6(model)
		model = self.bn_d6(model)
		model = self.relu_d6(model)

		#dp 7
		model = self.dc2d_d7(model)
		model = self.bn_d7(model)
		model = self.relu_d7(model)

		#dp 8
		model = self.dc2d_d8(model)
		model = self.bn_d8(model)
		model = self.relu_d8(model)

		#dp 9
		model = self.dc2d_d9(model)
		model = self.bn_d9(model)
		model = self.relu_d9(model)

		#dp 10
		model = self.dc2d_d10(model)
		model = self.bn_d10(model)
		model = self.relu_d10(model)

		#dp 11
		model = self.dc2d_d11(model)
		model = self.bn_d11(model)
		model = self.relu_d11(model)

		#dp 12
		model = self.zp_d12(model)
		model = self.dc2d_d12(model)
		model = self.bn_d12(model)
		model = self.relu_d12(model)

		#dp 13
		model = self.dc2d_d13(model)
		model = self.bn_d13(model)
		model = self.relu_d13(model)

		model = self.glavgpool2d_f(model)
		#model = self.res1_f(model)
		model = self.den_128(model)
		# model = self.drp_f(model)
		# model = self.conv2d_f(model)
		return self.norm_f(model)

if "__name__"=="__main__":
	model = BaseModel()
	images = tf.keras.layers.Input(name ="Input", shape = [96, 96, 3], dtype = tf.float32)
	embeddings = model.call(images)
	siamese_model = tf.keras.Model(inputs = images, outputs = embeddings)
	siamese_model.compile(optimizer = tf.keras.optimizers.Adam(0.001), loss = tfa.losses.TripletHardLoss(margin = 0.2))
	siamese_model.summary()
