import numpy as np


def GAE_advantages_and_returns(values, next_values, rewards_list, dones_list, gamma, lamda):
	'''
	values, next_values: (t_step,) np. array with the values of states
	rewards: list of size t_step 
	dones: list of size t_step

	'''
	t_steps = values.shape[0]
	rewards = np.array(rewards_list) #(t_step,)
	dones = np.array(dones_list) #(t_step,)
	lamda_s = 0
	lamda_advantages = []

	for t in range(t_steps - 1, -1, -1):
		td_error = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]

		lamda_s = td_error + gamma * lamda * lamda_s * (1 - dones[t])
		lamda_advantages.append(lamda_s)
	lamda_advantages.reverse()


	returns = np.array(lamda_advantages) + values

	return returns, np.array(lamda_advantages)






#https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
class RunningMeanStd(object):
	# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
	def __init__(self, epsilon=1e-4, shape=(), dtype=np.float32):
		self.mean = np.zeros(shape, dtype=dtype)
		self.var = np.ones(shape, dtype=dtype)
		self.count = epsilon

	def update(self, x):
		batch_mean = np.mean(x, axis=0, keepdims=True)
		batch_var = np.var(x, axis=0, keepdims=True)
		batch_count = x.shape[0]
		return self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean, batch_var, batch_count):
		delta = batch_mean - self.mean
		tot_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / tot_count
		m_a = self.var * (self.count)
		m_b = batch_var * (batch_count)
		M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
		new_var = M2 / (self.count + batch_count)

		new_count = batch_count + self.count

		self.mean = new_mean
		self.var = new_var
		self.count = new_count
		return self.mean, np.sqrt(self.var)


# https://github.com/openai/random-network-distillation
class RewardForwardFilter(object):   
	def __init__(self, gamma):
		self.rewems = None
		self.gamma = gamma
	def update(self, rews):
		if self.rewems is None:
			self.rewems = rews
		else:
			self.rewems = self.rewems * self.gamma + rews
		#print(self.rewems)
		return self.rewems 
