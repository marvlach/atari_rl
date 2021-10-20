import numpy as np
import random

class ExperienceReplayMemory(object):
	def __init__(self, size=1000000, obs_stack=4):
		self.observations = []
		self.actions = []
		self.rewards = []
		self.dones = []
		self.obs_stack = obs_stack
		self.size = size
		self.next_idx = 0

	def store(self, obs, action, reward, done):
		# being in previous state, we took action, received reward and observation
		# and now we are in new state which is final/done
		if self.next_idx >= len(self.observations):       
			self.observations.append(obs)
			self.actions.append(action)
			self.rewards.append(reward)
			self.dones.append(done)
		else:
			self.observations[self.next_idx] = obs
			self.actions[self.next_idx] = action
			self.rewards[self.next_idx] = reward
			self.dones[self.next_idx] = done # 
		self.next_idx = (self.next_idx + 1) % self.size




	def _get_slice(self, a, low, high):
        
		if low < 0:
			return a[low:] + a[:high]    
		else:
			return a[low:high]



		
	def sample_minibatch(self, batch_size):
		if len(self.observations) < batch_size + self.obs_stack + 1:
    			raise ValueError('Not enough memories to get a minibatch')

		idx_list = self._get_idx(batch_size)
		states_list = []
		new_states_list = []
		rewards_list = []
		actions_list = []
		dones_list = []
		for i, idx in enumerate(idx_list):
			state = self._get_slice(self.observations, idx - self.obs_stack, idx) #list of (x, x, 1) arrays
			new_state = self._get_slice(self.observations, idx - self.obs_stack + 1, idx + 1)# list of  (x, x, 1) arrays

			states_list.append(np.concatenate(state, axis = state[0].ndim-1)[None, :])# list of (1, x, x, obs_stack) arrays
			new_states_list.append(np.concatenate(new_state, axis = new_state[0].ndim-1)[None, :])# list of (1, x, x, obs_stack) arrays
			rewards_list.append(self.rewards[idx])
			actions_list.append(self.actions[idx])
			dones_list.append(self.dones[idx])
		return states_list, actions_list, rewards_list, new_states_list, dones_list

	def get_all_indices(self):
		all_indices = []
		for index in range(len(self.observations)):
			if len(self.observations) == self.size and index >= self.next_idx and index < self.next_idx + self.obs_stack \
				or len(self.observations) < self.size and index < self.obs_stack:
				continue

			# if any of previous state_length frames are terminal
			#print(self._get_slice(self.dones, index - self.obs_stack, index))
			if True in self._get_slice(self.dones, index - self.obs_stack, index): 
				continue

			# no replacement
			if index in all_indices:
				continue  
 
			all_indices.append(index)

		return all_indices 


	def _get_idx(self, batch_size):
		idx_list = []
		for i in range(batch_size):
			while True:
				index = random.randint(0, len(self.observations) - 1)
				# if mem. not full index must be > self.state_length
				# if mem. is full index in [0, self.current)U[self.current + self.state_length, size] 
				if len(self.observations) == self.size and index >= self.next_idx and index < self.next_idx + self.obs_stack \
					or len(self.observations) < self.size and index < self.obs_stack:
					continue

				# if any of previous state_length frames are terminal
				#print(self._get_slice(self.dones, index - self.obs_stack, index))
				if True in self._get_slice(self.dones, index - self.obs_stack, index): 
					continue

				# no replacement
				if index in idx_list:
					continue

				break
			idx_list.append(index)
		return idx_list




class PrioritizedExperienceReplayMemory(object):
	def __init__(self, size=1000000, obs_stack=4, alpha=1):
		self.observations = []
		self.actions = []
		self.rewards = []
		self.dones = []
		self.obs_stack = obs_stack
		self.size = size
		self.alpha = alpha

		tree_cap = 1
		while tree_cap < self.size:
			tree_cap *= 2

		self.tree_size = tree_cap #closest power of 2>size
		self.tree = [{'sum': 0, 'min':float('inf')} for _ in range(2 * self.tree_size)] # index=1 is root, tree has 2^n - 1 nodes

		self.max_priority = 1

		self.next_idx = 0


	def store(self, obs, action, reward, done, print_proc):
		# being in previous state, we took action, received reward and observation
		# and now we are in new state which is final/done
		if self.next_idx >= len(self.observations):       
			self.observations.append(obs)
			self.actions.append(action)
			self.rewards.append(reward)
			self.dones.append(done)
		else:
			self.observations[self.next_idx] = obs
			self.actions[self.next_idx] = action
			self.rewards[self.next_idx] = reward
			self.dones[self.next_idx] = done # 

		# put max priority in tree O(obs_stack*logn)
		idx = self.next_idx
		idx += self.tree_size #idx points at leaf
		self.tree[idx]['sum'] = self.max_priority ** self.alpha
		self.tree[idx]['min'] = self.max_priority ** self.alpha

		#update all parents
		idx //= 2
		while idx >= 1:
			self.tree[idx]['sum'] = self.tree[2*idx]['sum'] + self.tree[2*idx + 1]['sum']
			self.tree[idx]['min'] = min(self.tree[2*idx]['min'], self.tree[2*idx + 1]['min'])           
			idx //= 2
		######

		###########################################################################
		# do the same with 
		'''idx = self.next_idx + self.tree_size
		for i in range(idx + 1, (idx + 1 + self.obs_stack) % (2 *self.tree_size)):
			self.tree[i]['sum'] = 0
			self.tree[i]['min'] = float('inf')
			j = i//2
			while j >= 1:
				self.tree[j]['sum'] = self.tree[2*j]['sum'] + self.tree[2*j + 1]['sum']
				self.tree[j]['min'] = min(self.tree[2*j]['min'], self.tree[2*j + 1]['min'])           
				j //= 2'''

		###########################################################################


		if print_proc:
			print('a new frame in memory at index ' + str(self.next_idx) + ', tree_index ' + str(self.next_idx + self.tree_size) + ', priority' + str(self.tree[self.next_idx + self.tree_size ]['sum']))
		######
		

		#next idx
		self.next_idx = (self.next_idx + 1) % self.size



	def _get_slice(self, a, low, high):

		if low < 0:
			return a[low:] + a[:high]    
		else:
			return a[low:high]



	def sample_minibatch(self, batch_size, beta, print_proc):

		if len(self.observations) < batch_size + self.obs_stack + 1:
			raise ValueError('Not enough memories to get a minibatch')

		idx_list = self._get_idx(batch_size)


		p_min = self.tree[1]['min'] / self.tree[1]['sum']
		max_weight = (p_min * self.size) ** (-beta)######self.size
		#print([self.tree[1]['min'] , min([self.tree[i]['min'] for i in range(self.tree_size, 2*self.tree_size)]), max([self.tree[i]['sum'] for i in range(self.tree_size, 2*self.tree_size)])])
		states_list = []
		new_states_list = []
		rewards_list = []
		actions_list = []
		dones_list = []
		weights_list = []
		for i, idx in enumerate(idx_list):

			p_sample = self.tree[idx + self.tree_size]['sum'] / self.tree[1]['sum']
			weight = (p_sample * self.size) ** (-beta) ###############self.size


			weights_list.append(weight / max_weight)
			#weights_list.append(weight)


			#print([p_sample, p_min, self.tree[idx + self.tree_size]['sum'], self.tree[1]['min'], weight, max_weight])
			state = self._get_slice(self.observations, idx - self.obs_stack, idx) #list of (x, x, 1) arrays
			new_state = self._get_slice(self.observations, idx - self.obs_stack + 1, idx + 1)# list of  (x, x, 1) arrays

			states_list.append(np.concatenate(state, axis = state[0].ndim-1)[None, :])# list of (1, x, x, obs_stack) arrays
			new_states_list.append(np.concatenate(new_state, axis = new_state[0].ndim-1)[None, :])# list of (1, x, x, obs_stack) arrays
			rewards_list.append(self.rewards[idx])
			actions_list.append(self.actions[idx])
			dones_list.append(self.dones[idx])
			if print_proc:
				print('sampled index '+ str(idx) + ', with priority' + str(self.tree[idx + self.tree_size]['sum']) + ' and weight ' + str(weight / max_weight))
			'''max_weight = max(weights_list)
			weights_list = [weight/max_weight for weight in weights_list]
			if print_proc:
			print('weights list')
			print(weights_list)'''
		return states_list, actions_list, rewards_list, new_states_list, dones_list, weights_list, idx_list


	def retrieve(self, s):
		'''Find the highest index `i` in the array such that
		sum(arr[0] + arr[1] + ... + arr[i - i]) <= s'''

		idx = 1
		while idx < self.tree_size: #while non-leaf
			if self.tree[2 * idx]['sum'] > s:
				idx = 2 * idx
			else: 
				s -= self.tree[2 * idx]['sum'] 
				idx = 2 * idx + 1
		return idx - self.tree_size



	def _get_idx(self, batch_size):
		idx_list = []
		p_total = self.tree[1]['sum']
		sub_range = p_total / batch_size

		for i in range(batch_size):
			while True:
				#index = random.randint(0, len(self.observations) - 1)
				mass = random.random() * sub_range + i * sub_range
				index = self.retrieve(mass)

				if index >= len(self.observations):
					continue

				# if mem. not full, index must be > self.state_length
				# if mem. is full, index in [0, self.current)U[self.current + self.state_length, size] 
				if len(self.observations) == self.size and index >= self.next_idx and index < self.next_idx + self.obs_stack \
					or len(self.observations) < self.size and index < self.obs_stack:
					continue

				# if any of previous state_length frames are terminal
				#print(self._get_slice(self.dones, index - self.obs_stack, index))
				if True in self._get_slice(self.dones, index - self.obs_stack, index): 
					continue

				# no replacement
				if index in idx_list:
					continue

				break
			#print('sampled index '+ str(index) + ', with priority' + str(self.tree[index + self.tree_size]['sum']))
			idx_list.append(index)

		return idx_list

	def update_priorities(self, indices, priorities, print_proc):
		#print(indices)
		for idx, priority in zip(indices, priorities):
			#print(priority)
			tree_idx = idx + self.tree_size #idx points at leaf
			self.tree[tree_idx]['sum'] = priority ** self.alpha
			self.tree[tree_idx]['min'] = priority ** self.alpha

			#update all parents
			tree_idx //= 2
			while tree_idx >= 1:
				self.tree[tree_idx]['sum'] = self.tree[2*tree_idx]['sum'] + self.tree[2*tree_idx + 1]['sum']
				self.tree[tree_idx]['min'] = min(self.tree[2*tree_idx]['min'], self.tree[2*tree_idx + 1]['min'])           
				tree_idx //= 2

			self.max_priority = max(self.max_priority, priority)
			if print_proc:
				print('updated '+str(idx) + ' with new priority '+str(priority))


