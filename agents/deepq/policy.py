import numpy as np

class EpsGreedyPolicy(object):
	def __init__(self, action_size):
		self.action_size = action_size

	def get_action(self, eps, logits):
		#print([logits, np.argmax(logits, axis=-1)])
		if np.random.rand(1) < eps:
			return np.random.randint(0, self.action_size-1)
		else:
			return np.argmax(logits, axis=-1)

