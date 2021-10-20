import tensorflow as tf
import numpy as np



class ActorCriticModel(tf.keras.Model):
	def __init__(self, n_actions):
		super().__init__(name='ac')
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(8, 8), strides=4, input_shape=(84, 84, 4), data_format='channels_last',
							activation='relu', padding="valid",
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))
							#kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							#bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, 
							activation='relu', padding="valid",
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))
							#kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							#bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, 
							activation='relu', padding="valid",
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))
							#kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
                                                        #bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		self.flatten = tf.keras.layers.Flatten()
		self.dense = tf.keras.layers.Dense(512, activation='relu',
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))
							#kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							#bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		'''self.policy_dense = tf.keras.layers.Dense(512, activation='relu',
							kernel_initializer=tf.keras.initializers.Orthogonal(0.1))
		self.value_dense = tf.keras.layers.Dense(512, activation='relu',
							kernel_initializer=tf.keras.initializers.Orthogonal(0.1))'''
		self.policy_logits = tf.keras.layers.Dense(n_actions, activation='linear',
							kernel_initializer=tf.keras.initializers.Orthogonal(0.1))
							#kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							#bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))

		self.ext_value = tf.keras.layers.Dense(1, activation='linear',
							kernel_initializer=tf.keras.initializers.Orthogonal(0.1))
							#kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							#bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))


	def call(self, inputs):
		x = inputs/255.
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense(x)
		'''pol_dense = self.policy_dense(x)
		val_dense = self.value_dense(x)
		logits = self.policy_logits(pol_dense)
		ext_values = self.ext_value(val_dense)'''
		logits = self.policy_logits(x)
		#int_values = self.int_value(x)
		ext_values = self.ext_value(x)
		return logits, ext_values





class RNDPredictorModel(tf.keras.Model):
	def __init__(self):
		super().__init__(name='pred')
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(8, 8), strides=4, input_shape=(84, 84, 4), data_format='channels_last',
							activation=tf.nn.leaky_relu, padding="valid",
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))#tf.keras.initializers.VarianceScaling(scale=2.0)
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, 
							activation=tf.nn.leaky_relu, padding="valid",
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, 
							activation=tf.nn.leaky_relu, padding="valid", 
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))
		self.dense2 = tf.keras.layers.Dense(512, activation='linear', kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))

	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)
		return x


class RNDTargetModel(tf.keras.Model):
	def __init__(self):
		super().__init__(name='rnd')
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(8, 8), strides=4, input_shape=(84, 84, 4), data_format='channels_last',
							activation=tf.nn.leaky_relu, padding="valid",
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)), trainable=False)
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, 
							activation=tf.nn.leaky_relu, padding="valid",
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)), trainable=False)
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, 
							activation=tf.nn.leaky_relu, padding="valid", #########tf.keras.layers.LeakyReLU(alpha=0.1)
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)), trainable=False)
		self.flatten = tf.keras.layers.Flatten(trainable=False)
		self.dense = tf.keras.layers.Dense(512, activation='linear', kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)), trainable=False)

	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense(x)
		return x


class ICMModel(tf.keras.Model):
	def __init__(self, n_actions):
		super().__init__(name='icm')
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(8, 8), strides=4, input_shape=(84, 84, 4), data_format='channels_last',
							activation=tf.nn.leaky_relu, padding="valid",
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, 
							activation=tf.nn.leaky_relu, padding="valid",
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, 
							activation=tf.nn.leaky_relu, padding="valid", 
							kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))
		self.flatten = tf.keras.layers.Flatten()
		self.features = tf.keras.layers.Dense(512, activation='linear', kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))

		self.inverse_dense = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))       
		self.inverse = tf.keras.layers.Dense(n_actions, activation='linear', kernel_initializer=tf.keras.initializers.Orthogonal(0.1))  

		self.forward_dense = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))       
		self.forward = tf.keras.layers.Dense(512, activation='linear', kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.0)))


	def call(self, inputs):
		state, next_state, action = inputs # action is in one-hot (batch, n_actions)
		# features for state and next state
		state_code = self.features(self.flatten(self.conv3(self.conv2(self.conv1(state))))) # (batch, code)
		next_state_code = self.features(self.flatten(self.conv3(self.conv2(self.conv1(next_state))))) # (batch, code)

		# pred action from inverse
		pred_action = self.inverse(self.inverse_dense(tf.concat([state_code, next_state_code], axis=-1)))

		# pred next state code
		pred_next_state_code = self.forward(self.forward_dense(tf.concat([state_code, action], axis=-1)))

		return next_state_code, pred_next_state_code, pred_action
