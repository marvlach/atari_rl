
import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import tensorflow as tf
from memory import Memory

class Worker(threading.Thread):
	# Set up global variables across different threads
	global_episode = 0   
	global_steps = 0 
	save_lock = threading.RLock()
	#global_steps = 0

	def __init__(self, state_shape, action_size, env_wrapper, 
			frame_stack, t_steps, n_steps, max_episode_steps, reward_clip, 
			reward_queue, step_queue, data_queue, term_queue, action_queue, current_state_queue,
			barrier_lock, idx, game_name, save_dir):
		super(Worker, self).__init__()
		self.state_shape = state_shape
		self.action_size = action_size

		self.game_name = game_name
		self.env_wrapper = env_wrapper
		if self.env_wrapper is not None:# for atari
			self.env = self.env_wrapper(gym.make(self.game_name))
		else: # 
			self.env = gym.make(self.game_name)   

		self.frame_stack = frame_stack

		self.n_steps = n_steps
		self.t_steps = t_steps
		self.max_episode_steps = max_episode_steps
		self.reward_clip = reward_clip

		self.reward_queue = reward_queue
		self.step_queue = step_queue
		self.data_queue = data_queue
		self.action_queue = action_queue
		self.current_state_queue = current_state_queue
		self.term_queue = term_queue

		self.barrier_lock = barrier_lock
		self.worker_idx = idx
		self.save_dir = save_dir

	def run(self):
		mem = Memory()
		current_obs = self.env.reset()
		if self.frame_stack > 1: #atari equiv. to if current_obs.ndim == 22
			current_state = np.repeat(np.expand_dims(current_obs, axis=2), self.frame_stack, axis = 2)   
		else:
			current_state = current_obs

		ep_reward = 0.0
		ep_steps = 0
		# flag given by agent
		term = False

		while not term:

			# collect a rollout
			for t in range(self.t_steps): 

				# send current state to agent and wait for action
				self.current_state_queue.put(current_state[None, :])
				action = self.action_queue.get()

				#perform action
				new_obs, reward, done_life, done_ep, _ = self.env.step(action)
				#stack new obs to current state
				if self.frame_stack > 1:
					new_state = np.append(current_state[:, :, 1:], np.expand_dims(new_obs, axis=2), axis=2)
				else:
					new_state = new_obs

				ep_reward += reward
				ep_steps += 1
				clipped_reward = reward
				if self.reward_clip:
					clipped_reward = np.sign(reward)
				# store transition to memory
				mem.store(current_state[None, :], action, clipped_reward, new_state[None, :], done_life or ep_steps >= self.max_episode_steps)



				# end of episode
				if done_life or ep_steps >= self.max_episode_steps:
					# put episode info in queue
					with Worker.save_lock:
						self.reward_queue.put(ep_reward)
						self.step_queue.put(ep_steps)   

					# reset environment
					current_obs = self.env.reset()
					if self.frame_stack > 1: #atari equiv. to if current_obs.ndim == 22
						current_state = np.repeat(np.expand_dims(current_obs, axis=2), self.frame_stack, axis = 2)   
					else:
						current_state = current_obs

					# reset episode info
					ep_reward = 0.0
					ep_steps = 0
				# not end of episode
				else:
					current_state = new_state

			#send rollout
			t_step_data = {'memory' : mem, 'idx' : self.worker_idx}
			with Worker.save_lock:
				self.data_queue.put(t_step_data)

			# wait for update
			# we should block here until master thread updates the network
			self.barrier_lock.wait()

			# check if master said to terminate
			with Worker.save_lock: 
				if not self.term_queue.empty():
					term = self.term_queue.get()
			# clear memory
			mem.clear()

		#end of training
		self.env.close()
		self.reward_queue.put(None)
		self.step_queue.put(None)
		with Worker.save_lock:
			print('end of training')
