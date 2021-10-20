import tensorflow as tf


class DQN(tf.keras.Model):
	def __init__(self, n_actions):
		super().__init__(name='dqn')
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(8, 8), strides=4, input_shape=(84, 84, 4),data_format='channels_last',
							activation='relu', padding="valid",
							kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, 
							activation='relu', padding="valid",
							kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, 
							activation='relu', padding="valid",
							kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		self.flatten = tf.keras.layers.Flatten()
		self.dense = tf.keras.layers.Dense(512, activation='relu',
							kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							bias_initializer=tf.keras.initializers.Constant(0.0))
		self.q_values = tf.keras.layers.Dense(n_actions, activation='linear')   

	@tf.function            
	def call(self, inputs):
		x = inputs/255.
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		x = self.dense(x)
		Q = self.q_values(x)
		return Q

class duelDQN(tf.keras.Model):
	def __init__(self, n_actions):
		super().__init__(name='dueldqn')
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(8, 8), strides=4, input_shape=(84, 84, 4), data_format='channels_last',
							activation='relu', padding="valid",
							kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, 
							activation='relu', padding="valid",
							kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, 
							activation='relu', padding="valid",
							kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		self.flatten = tf.keras.layers.Flatten()
		self.dense_adv = tf.keras.layers.Dense(512, activation='relu',
							kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		self.dense_val = tf.keras.layers.Dense(512, activation='relu',
							kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
							bias_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
		self.advantages = tf.keras.layers.Dense(n_actions, activation='linear')   
		self.value = tf.keras.layers.Dense(1, activation='linear')   
        
	@tf.function            
	def call(self, inputs):
		x = inputs/255.
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		val = self.dense_val(x)
		adv = self.dense_adv(x)
		val = self.value(val)
		adv = self.advantages(adv)
		Q = tf.math.add(val, tf.math.subtract(adv, tf.math.reduce_mean(adv, axis=-1, keepdims=True)))
		#Q = tf.math.add(val, tf.math.subtract(adv, tf.math.reduce_max(adv, axis=-1, keepdims=True)))
		return Q

