# i was in state, i took an action, saw a reward and now i am in a new state which is done
class Memory(object):
	def __init__(self):
		self.states = []
		self.new_states = []
		self.actions = []
		self.rewards = []
		self.dones = []

	def store(self, state, action, reward, new_state, done):
		self.states.append(state)
		self.new_states.append(new_state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.dones.append(done)
	def clear(self):
		self.states = []
		self.new_states = []
		self.actions = []
		self.rewards = []
		self.dones = []
