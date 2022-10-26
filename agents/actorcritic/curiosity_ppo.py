import sys
import os
import tensorflow as tf
import numpy as np
import random
import gym
import csv
from gym.wrappers.record_video import RecordVideo
from queue import Queue
import threading
from worker import Worker
from models import ActorCriticModel, RNDPredictorModel, RNDTargetModel, ICMModel

from utils import GAE_advantages_and_returns, RunningMeanStd, RewardForwardFilter




class CuriosityActorCriticAgent():
	def __init__(self, agent_name, game_name, save_dir, env_wrapper, frame_stack, curiosity_module=None):
		self.game_name = game_name
		self.agent_name = agent_name
		self.save_dir = save_dir

		self.env_wrapper = env_wrapper
		self.frame_stack = frame_stack
		self.curiosity_module = curiosity_module
		assert curiosity_module in ['rnd', 'icm'], 'curiosity must be rnd or icm'

		if self.env_wrapper is not None:# for atari
			env = self.env_wrapper(gym.make(self.game_name))
			self.test_env = self.env_wrapper(
				RecordVideo(gym.make(self.game_name), 
				os.path.join(self.save_dir, self.agent_name, self.game_name, 'video'),  
				episode_trigger = lambda episode_number: episode_number % 10 == 0)
			)
		else:
			env = gym.make(self.game_name)
			self.test_env = RecordVideo(
				gym.make(self.game_name), 
				os.path.join(self.save_dir, self.agent_name, self.game_name, 'video'),
				episode_trigger = lambda episode_number: episode_number % 10 == 0
			)
		self.obs_shape = env.observation_space.shape

		# assert that observations are images
		assert len(self.obs_shape) == 2, "Observations must be grayscale images"

		if frame_stack > 1:#Atari (x, x, frame_stack)
			self.state_shape = self.obs_shape + (self.frame_stack, )
		else: #
			self.state_shape = self.obs_shape

		self.action_size = env.action_space.n


		self.ext_gamma = 0.99 ##############################################################0.999
		self.int_gamma = 0.99
		self.lamda = 0.95
		self.grad_clip = 0.5

		self.global_model = ActorCriticModel(self.action_size)  # global actor-critic network trainable 

		self.test_model = ActorCriticModel(self.action_size)  # network used for evaluation non-trainable
		self.test_model(tf.convert_to_tensor(np.random.random((1,) + self.state_shape), dtype=tf.float32))

		self.ext_val_coeff = 0.5
		self.ext_adv_coeff = 1


		#for rnd we use a predictor network that we train on the outputs of a randomly initialized target network
		if curiosity_module == 'rnd':
			self.rnd_target = RNDTargetModel() # target network non-trainable 
			self.rnd_predictor = RNDPredictorModel() # predictor network trainable(learns taget)

		# for icm we use a forward and an inverse model that share a CNN embedding
		elif curiosity_module == 'icm':
			self.icm_model = ICMModel(self.action_size) 


	def evaluate(self, best_avg100_reward, n_eval_episodes):
		# set test model, global model keeps updating by threads in parallel
		self.test_model.set_weights(self.global_model.get_weights())
		print('Evaluating for {} episodes...'.format(n_eval_episodes))
		test_steps = []
		test_rewards = []


		for ep in range(n_eval_episodes):
			ep_rew, ep_step = self.play(self.test_env)
			#acc_test_steps += ep_steps
			test_rewards.append(ep_rew)
			test_steps.append(ep_step)
			print('Evaluation episode: '+str(ep)+', reward: '+str(ep_rew)+', steps: '+str(ep_step))

		# compute 100-episode avg reward
		avg100_reward = np.mean(test_rewards)
		print('Evaluation completed with avg. reward: '+str(avg100_reward))
		# if better than previous, save model
		if avg100_reward >= best_avg100_reward:
			best_avg100_reward = avg100_reward
			self.test_model.save_weights(os.path.join(self.save_dir, self.agent_name, self.game_name, 'best_model.h5'))

		return best_avg100_reward


	def play(self, env, model_dir = None):
		# setup model
		model = self.test_model
		# used after training
		if model_dir:
			model_path = os.path.join(self.save_dir, self.agent_name, self.game_name, 'best_model.h5')
			print('Loading model from: {}'.format(model_path))
			model.load_weights(model_path)

		# used during traiing
		#else:
			#print('Using current test model')


		done = False
		ep_steps = 0
		ep_reward = 0
		current_obs = env.reset()

		if self.frame_stack > 1: #atari equiv. to if current_obs.ndim == 
			current_state = np.repeat(np.expand_dims(current_obs, axis=2), self.frame_stack, axis = 2)   
		else:
			current_state = current_obs


		while not done:
			#env.render(mode='rgb_array')
			logits, value = model(tf.convert_to_tensor(current_state[None, :], dtype=tf.float32))
			policy = tf.nn.softmax(logits)
			#action = np.argmax(policy)
			#print(policy.numpy()[0])
			action = np.random.choice(env.action_space.n, p=policy.numpy()[0])
			new_obs, reward, done, _, _ = env.step(action)

			if self.frame_stack > 1:
				new_state = np.append(current_state[:, :, 1:], np.expand_dims(new_obs, axis=2), axis=2)
			else:
				new_state = new_obs

			ep_reward += reward
			ep_steps += 1
			current_state = new_state

		return ep_reward, ep_steps


	def get_rollout(self, n_threads, t_steps):


		# pick actions for threads for the entire rollout
		for step in range(t_steps):
			thread_states = [] #list of arrays (1,x,x,4)

			# get state of every thread
			for thread in range(n_threads):
				thread_states.append(self.thread_state_queues[thread].get())  

			# pick an action
			logits, _ = self.global_model(tf.convert_to_tensor(np.concatenate(thread_states, axis=0), dtype=tf.float32))
			probs = tf.nn.softmax(logits).numpy()
			thread_actions = [np.random.choice(self.action_size, p=thread_dist) for thread_dist in probs]

			# send action to every thread
			for thread in range(n_threads):
				self.action_queues[thread].put(thread_actions[thread])


		batch_states_list = []
		batch_next_states_list = []
		batch_actions_list = []
		batch_ext_rewards_list = []
		batch_dones_list = []

		batch_int_rewards_list = []
		batch_int_invret_list = []

		# get t-step rollout from each thread
		for i in range(n_threads):
			# experience [idx] corresponds to i-th thread
			data_dict = self.data_queue.get()
			data_idx = data_dict['idx']
			batch_states_list.append(data_dict['memory'].states) #list of lists (n_threads, t_steps) of arrays (1,x,x,4)
			batch_next_states_list.append(data_dict['memory'].new_states) #list of lists (n_threads, t_steps) of arrays (1,x,x,4)
			batch_actions_list.append(data_dict['memory'].actions) #list of lists (n_threads, t_steps) of ints
			batch_ext_rewards_list.append(data_dict['memory'].rewards) #list of lists (n_threads, t_steps) of floats
			batch_dones_list.append(data_dict['memory'].dones) #list of lists (n_threads, t_steps) of bool

			if not True in data_dict['memory'].dones:
				assert not False in [np.array_equal(data_dict['memory'].states[s+1], data_dict['memory'].new_states[s]) for s in range(len(data_dict['memory'].states)-1)]
			'''print('episode ended?')
			print(True in data_dict['memory'].dones)'''
			# calculate intrisic rewards: 
			rollout_states = np.concatenate(data_dict['memory'].states, axis=0) #(t,84,84,4)
			rollout_next_states = np.concatenate(data_dict['memory'].new_states, axis=0) #(t,84,84, 4)

			rollout_int_rewards_list, rollout_int_invret_list = self.calculate_intrinsic_rewards(rollout_states, rollout_next_states, 
												data_dict['memory'].actions, data_idx) #list of lists of floats


			batch_int_rewards_list.append(rollout_int_rewards_list) #list of lists of floats
			batch_int_invret_list.append(rollout_int_invret_list) #list of lists of floats


		return batch_states_list, batch_next_states_list, batch_actions_list, batch_ext_rewards_list, batch_dones_list, batch_int_rewards_list, batch_int_invret_list


	def get_training_data(self, batch_states_list, batch_next_states_list, batch_actions_list, batch_ext_rewards_list, batch_int_rewards_list, batch_dones_list):
		# lists of data points
		batch_total_returns = []
		batch_states = []
		batch_next_states = []
		batch_actions = []
		batch_total_advantages = []
		batch_next_observations = []

		#update#
		# for every thread: get returns and advantages and make one big batch
		for (states_list, next_states_list, actions, ext_rewards, int_rewards, dones_list) in zip(batch_states_list, batch_next_states_list, batch_actions_list, 
													batch_ext_rewards_list, batch_int_rewards_list, batch_dones_list): ########

			# values and bootstrap value               
			states = np.concatenate(states_list, axis=0)
			_, values = self.global_model(tf.convert_to_tensor(states, dtype=tf.float32))#.numpy() #(t, 1)
			ext_values = values.numpy() # (t,1)

			next_states = np.concatenate(next_states_list, axis=0)
			_, next_values = self.global_model(tf.convert_to_tensor(next_states, dtype=tf.float32))#.numpy() #(t, 1)
			ext_next_values = next_values.numpy() # (t,1)

			# normalize intrinsic rewards
			int_rewards /= np.sqrt(self.int_invret_running.var)
			int_rewards = int_rewards.tolist()


			self.int_rewards_list_debug.extend(int_rewards)

			# total reward
			total_rewards = 0.05 * np.array(int_rewards) + np.array(ext_rewards)

			# get returns and advantages
			total_returns, total_advantages = GAE_advantages_and_returns(np.ravel(ext_values), np.ravel(ext_next_values), total_rewards.tolist(), dones_list, self.ext_gamma, self.lamda)


			# make lists of training data
			batch_total_returns += total_returns.tolist()
			batch_states += states_list
			batch_next_states += next_states_list
			batch_actions += actions
			batch_total_advantages += total_advantages.tolist()
			batch_next_observations += [np.expand_dims(next_states_list[j][:, :, :, -1], axis=-1) for j in range(len(next_states_list))] #list of (1, x, x, 1)

		# normalize advantages
		'''total_advantages_array = np.array(batch_total_advantages)
		norm_total_advantages = (total_advantages_array - np.mean(total_advantages_array)) / np.std(total_advantages_array)
		batch_total_advantages = norm_total_advantages.tolist()
		self.total_norm_advanatages_list_debug += batch_total_advantages'''

		return batch_total_returns, batch_states, batch_next_states, batch_actions, batch_total_advantages, batch_next_observations 


	def perform_update(self, n_opt_epochs, n_minibatches_per_epoch, 
			batch_states, batch_actions, batch_total_returns, batch_total_advantages, 
			batch_norm_states, batch_norm_next_states, batch_old_policy):


		batch_value_loss, batch_policy_loss, batch_entropy, batch_clipped_fraction = [], [], [], []
		batch_distillation_loss, batch_forward_loss, batch_inverse_loss, batch_icm_loss = [], [], [], []
		# every epoch iterate over the whole batch once in n_minibatches_per_epoch
		for epoch_i in range(n_opt_epochs):

			# shuffle batch
			experiences = list(zip(batch_states, batch_actions, batch_total_returns, batch_total_advantages, batch_norm_states, batch_norm_next_states, batch_old_policy))
			random.shuffle(experiences)
			batch_states, batch_actions, batch_total_returns, batch_total_advantages, batch_norm_states, batch_norm_next_states, batch_old_policy = zip(*experiences)
			batch_states = list(batch_states)
			batch_actions = list(batch_actions)
			batch_total_returns = list(batch_total_returns)
			batch_total_advantages = list(batch_total_advantages)
			batch_old_policy = list(batch_old_policy)
			batch_norm_states = list(batch_norm_states)
			batch_norm_next_states = list(batch_norm_next_states)

			# split batch into minibatches and perform training step for every minibatch
			for mb in range(n_minibatches_per_epoch):
				start = mb * len(batch_states)  // n_minibatches_per_epoch
				finish = (mb + 1) * len(batch_states)  // n_minibatches_per_epoch  

				# slice minibatch
				mb_states = batch_states[start:finish]
				mb_actions = batch_actions[start:finish]
				mb_total_returns = batch_total_returns[start:finish]
				mb_total_advantages = batch_total_advantages[start:finish]
				mb_norm_states = batch_norm_states[start:finish]   #[start:start + (finish - start) // int(1/curiosity_keep_prob)] # if we used more than 32 workers
				mb_norm_next_states = batch_norm_next_states [start:finish]  #[start:start + (finish - start) // int(1/curiosity_keep_prob)] # if we used more than 32 workers
				mb_old_policy = batch_old_policy[start:finish]




				# update actor-critic network with minibatch
				value_loss, policy_loss, entropy = self._update_network(tf.convert_to_tensor(np.array(mb_total_returns)[:, None], dtype=tf.float32),
											tf.convert_to_tensor(np.concatenate(mb_states, axis=0), dtype=tf.float32),
											tf.convert_to_tensor(np.array(mb_actions), dtype=tf.int32), 
											tf.convert_to_tensor(np.array(mb_total_advantages)[:, None], dtype=tf.float32),
											tf.convert_to_tensor(np.array(mb_old_policy), dtype=tf.float32))

				# stats
				batch_value_loss.append(value_loss.numpy()) 
				batch_policy_loss.append(policy_loss.numpy())  
				batch_entropy.append(entropy.numpy()) 


				if self.curiosity_module == 'rnd':
					curiosity_loss = self._update_rnd_network(tf.convert_to_tensor(np.concatenate(mb_norm_next_states, axis=0), dtype=tf.float32))
					batch_distillation_loss.append(curiosity_loss.numpy())
				else: #icm
					forward_loss, inverse_loss, icm_loss = self._update_icm_network(tf.convert_to_tensor(np.concatenate(mb_norm_states, axis=0), dtype=tf.float32),
												tf.convert_to_tensor(np.concatenate(mb_norm_next_states, axis=0), dtype=tf.float32),
												tf.convert_to_tensor(np.array(mb_actions), dtype=tf.int32))
					batch_forward_loss.append(forward_loss.numpy())
					batch_inverse_loss.append(inverse_loss.numpy())
					batch_icm_loss.append(icm_loss.numpy())


		if self.curiosity_module == 'rnd':
			with open(os.path.join(self.save_dir, self.agent_name, self.game_name, 'distillation_loss.csv'), 'a', newline='') as file:
				writer = csv.writer(file)
				writer.writerow([np.mean(batch_distillation_loss)])

		else:
			with open(os.path.join(self.save_dir, self.agent_name, self.game_name, 'forward_loss.csv'), 'a', newline='') as file:
				writer = csv.writer(file)
				writer.writerow([np.mean(batch_forward_loss)])

			with open(os.path.join(self.save_dir, self.agent_name, self.game_name, 'inverse_loss.csv'), 'a', newline='') as file:
				writer = csv.writer(file)
				writer.writerow([np.mean(batch_inverse_loss)])

			with open(os.path.join(self.save_dir, self.agent_name, self.game_name, 'icm_loss.csv'), 'a', newline='') as file:
				writer = csv.writer(file)
				writer.writerow([np.mean(batch_icm_loss)])



		# write results in csv
		with open(os.path.join(self.save_dir, self.agent_name, self.game_name, 'value_loss.csv'), 'a', newline='') as file:
			writer = csv.writer(file)
			writer.writerow([np.mean(batch_value_loss)])


		with open(os.path.join(self.save_dir, self.agent_name, self.game_name, 'policy_loss.csv'), 'a', newline='') as file:
			writer = csv.writer(file)
			writer.writerow([np.mean(batch_policy_loss)])

		with open(os.path.join(self.save_dir, self.agent_name, self.game_name, 'entropy.csv'), 'a', newline='') as file:
			writer = csv.writer(file)
			writer.writerow([np.mean(batch_entropy)])






	def train(self, t_steps, n_opt_epochs, n_minibatches_per_epoch, ppo_clip, entropy_coeff, n_steps, max_episode_steps, 
		reward_clip, n_threads, learning_rate, n_obs_init_rollouts, eval_epoch_steps, n_eval_episodes):
		self.reward_queue = Queue()
		self.step_queue = Queue()
		self.data_queue = Queue(maxsize=n_threads)
		self.term_queue = Queue(maxsize=n_threads)
		self.thread_state_queues = [Queue() for thread in range(n_threads)]
		self.action_queues = [Queue() for thread in range(n_threads)]
		self.barrier_lock = threading.Barrier(parties=n_threads + 1)


		self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1.5e-04)
		self.ppo_clip = ppo_clip
		self.entropy_coeff = entropy_coeff

		# for curiosity we keep a running mean, std of observations and 
		# a running mean, std of inverse intrinsic returns

		# seperate optimizer for curiosity module
		self.curiosity_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1.5e-04)

		# observation normalization params (1, x, x, 1)
		self.obs_running = RunningMeanStd(shape=(1,) + self.obs_shape + (1,))

		# intrinsic reward normalization params
		# for every thread we keep a seperate intrinsic inverse return
		self.int_invret_running = RunningMeanStd(shape=()) 
		self.int_invret_sum = [RewardForwardFilter(self.int_gamma) for _ in range(n_threads)]       


		# step a random agent to init. obs. norm. params
		init_env = self.env_wrapper(gym.make(self.game_name))
		obs_init_list = []
		init_steps = 0
		while init_steps < n_obs_init_rollouts * n_threads * t_steps:
			cur_obs = init_env.reset()
			done = False 
			obs_init_list.append(cur_obs[None, :, :, None])
			init_steps += 1

			while not done:
				action = np.random.choice(init_env.action_space.n)
				cur_obs, _, done, _, _ = init_env.step(action)
				'''print(cur_obs)
				input('hey')'''
				obs_init_list.append(cur_obs[None, :, :, None])    
				init_steps += 1
		new_mean, new_std = self.obs_running.update(np.concatenate(obs_init_list, axis=0))   

		del obs_init_list
		init_env.close()  

		self.int_rewards_list_debug = []
		self.value_loss_list_debug = []
		self.policy_loss_list_debug = []
		self.entropy_list_debug = []
		self.distillation_loss_list_debug = []
		self.forward_loss_list_debug = []
		self.inverse_loss_list_debug = []
		self.icm_loss_list_debug = []

		#start workers
		workers = [Worker(self.state_shape, self.action_size, self.env_wrapper,
				self.frame_stack, t_steps,  n_steps,  max_episode_steps,  reward_clip,
				self.reward_queue, self.step_queue, self.data_queue, self.term_queue, self.action_queues[i], self.thread_state_queues[i],
				self.barrier_lock, i, game_name=self.game_name, save_dir=self.save_dir) for i in range(n_threads)]


		for i, worker in enumerate(workers):
			print("Starting worker {}".format(i))
			worker.start()

		self.reward_list = []  # record episode reward to plot
		self.step_list = []

		best_avg100_reward = 0
		total_episode_steps = 0

		next_eval = eval_epoch_steps


		n_steps_reached = False
		# start training        
		while not n_steps_reached:

			# get rollout from all threads
			batch_states_list, batch_next_states_list, batch_actions_list, batch_ext_rewards_list, batch_dones_list, batch_int_rewards_list, batch_int_invret_list = \
					self.get_rollout(n_threads, t_steps)

			# update running mean, std of inverse intirinsic return
			self.int_invret_running.update(np.ravel(np.array(batch_int_invret_list)))
			print('mean')
			print(self.int_invret_running.mean)
			print('var')
			print(self.int_invret_running.var)
			print('rff')
			print([self.int_invret_sum[w].rewems for w in range(n_threads)])



			# all threads are blocked in barrier until update is done
			# make training data from experiences
			batch_total_returns, batch_states, batch_next_states, batch_actions, batch_total_advantages, batch_next_observations = \
					self.get_training_data(batch_states_list, batch_next_states_list, batch_actions_list, batch_ext_rewards_list, batch_int_rewards_list, batch_dones_list)


			# update next obs. normalization params 
			self.obs_running.update(np.concatenate(batch_next_observations, axis=0))

			# normalize states
			norm_states_array = np.clip((np.concatenate(batch_states, axis=0) - self.obs_running.mean) / np.sqrt(self.obs_running.var), -5, 5)
			batch_norm_states = [np.expand_dims(norm_states_array[sample,:,:,:], axis=0)   for sample in range(norm_states_array.shape[0])]

			# normalize next states
			norm_next_states_array = np.clip((np.concatenate(batch_next_states, axis=0) - self.obs_running.mean) / np.sqrt(self.obs_running.var), -5, 5)
			batch_norm_next_states = [np.expand_dims(norm_next_states_array[sample,:,:,:], axis=0)   for sample in range(norm_next_states_array.shape[0])]

			# get the pre-update old policy for every state (this won't be used for a2c)
			logits, _= self.global_model(tf.convert_to_tensor(np.concatenate(batch_states, axis=0), dtype=tf.float32))
			batch_old_policy = tf.nn.softmax(logits).numpy().tolist()



			#print(batch_norm_observations[-1][0,:,:,0])
			self.perform_update(n_opt_epochs, n_minibatches_per_epoch, batch_states, batch_actions, 
					batch_total_returns, batch_total_advantages, 
					batch_norm_states, batch_norm_next_states, batch_old_policy)


			#print('update is done')
			# check if an episode finished by checking reward queue
			while not self.reward_queue.empty() and not self.step_queue.empty():
				reward = self.reward_queue.get()
				step = self.step_queue.get()
				if reward is not None and step is not None:
					self.reward_list.append(reward)
					self.step_list.append(step)
					total_episode_steps += step
					print('global steps='+str(total_episode_steps)+', global episode='+str(len(self.reward_list))+', episode reward='+str(reward)+', episode steps='+str(step))
					print('last in reward=')
					print(self.int_rewards_list_debug[-100:-1:10])
			# time to evaluate
			if total_episode_steps > next_eval:
				best_avg100_reward = self.evaluate(best_avg100_reward, n_eval_episodes) 
				next_eval += eval_epoch_steps 

			# if we risk passing the limit of n_steps, send a termination message to threads and set flag for term.
			next_global_steps = total_episode_steps + n_threads*t_steps
			if next_global_steps >= n_steps:
				n_steps_reached = True
				for thr in range(n_threads):
					self.term_queue.put(True)

			# update and/or evaluation is done: unblock all threads
			self.barrier_lock.wait()


		[w.join() for w in workers]

		# write results in csv
		with open(os.path.join(self.save_dir, self.agent_name, self.game_name, 'learning_curve.csv'), 'w', newline='') as file:
			writer = csv.writer(file)
			for i in range(len(self.reward_list)):
				writer.writerow([i, self.step_list[i], self.reward_list[i]])


		# save final model weights
		self.global_model.save_weights(os.path.join(self.save_dir, self.agent_name, self.game_name, 'final_model.h5'))



		self.test_env.close()


	def calculate_intrinsic_rewards(self, states, next_states, actions, worker_idx):
		# states is (t, x, y, 4), actions is a list
		# convert actions to one-hot
		actions_one_hot = np.zeros((np.array(actions).size, self.action_size))
		actions_one_hot[np.arange(np.array(actions).size), np.array(actions)] = 1 # (t, actions_space)


		# normalize states and next_states
		norm_states = np.clip((states - self.obs_running.mean) / np.sqrt(self.obs_running.var), -5, 5)
		norm_next_states = np.clip((next_states - self.obs_running.mean) / np.sqrt(self.obs_running.var), -5, 5)

		if self.curiosity_module == 'rnd':
			# feed them to predictor and target model and get the L2-norm of difference(use mean instead of sum)
			target_embedding = self.rnd_target(tf.convert_to_tensor(norm_next_states, dtype=tf.float32)).numpy() #(rollout_length, features)
			predictor_embedding = self.rnd_predictor(tf.convert_to_tensor(norm_next_states, dtype=tf.float32)).numpy()  #(rollout_length, features)
			intrinsic_rewards = np.sum((target_embedding - predictor_embedding)**2, axis=-1).tolist()

		elif self.curiosity_module == 'icm':
			next_state_embedding, pred_next_state_embedding, _ = self.icm_model([tf.convert_to_tensor(norm_states, dtype=tf.float32), #(rollout_length, features)
											tf.convert_to_tensor(norm_next_states, dtype=tf.float32),
											tf.convert_to_tensor(actions_one_hot, dtype=tf.float32)])
			intrinsic_rewards = np.sum((next_state_embedding.numpy() - pred_next_state_embedding.numpy())**2, axis=-1).tolist()


		# calculate new inverse intrinsic return
		int_invret_list = []
		for int_rew in intrinsic_rewards:
			int_invret_list.append(self.int_invret_sum[worker_idx].update(int_rew)) #list of floats

		return intrinsic_rewards, int_invret_list



	def _update_rnd_network(self, mb_next_states):
		with tf.GradientTape() as tape:
			# Distillation Loss
			target_embedding = self.rnd_target(mb_next_states) #(mb_size, features)
			predictor_embedding = self.rnd_predictor(mb_next_states) #(mb_size, features)
			distillation_loss = tf.math.reduce_mean(tf.math.square(tf.stop_gradient(target_embedding) - predictor_embedding), axis=-1, keepdims=True) #(mb_size,1)
			total_loss = tf.reduce_mean(distillation_loss) 

		grads = tape.gradient(total_loss, self.rnd_predictor.trainable_weights)
		#grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
		grads_vars = list(zip(grads, self.rnd_predictor.trainable_weights))

		self.curiosity_optimizer.apply_gradients(grads_vars)

		return total_loss


	def _update_icm_network(self, mb_states, mb_next_states, mb_actions):
		#print(mb_observations.shape)
		with tf.GradientTape() as tape:
			action_chosen = tf.one_hot(mb_actions, self.action_size, axis=-1, dtype=tf.float32) # (mb_size, n_actions)
			# ICM
			next_state_embedding, pred_next_state_embedding, pred_action_logits = self.icm_model([mb_states, mb_next_states, action_chosen])
			pred_action = tf.nn.softmax(pred_action_logits)

			# forward loss
			forward_loss = tf.math.reduce_mean(tf.math.square(tf.stop_gradient(next_state_embedding) - pred_next_state_embedding), axis=-1, keepdims=True) #(mb_size,1)

			# inverse loss
			inverse_loss = -tf.math.reduce_sum(action_chosen * tf.math.log(pred_action + 1e-8), axis=-1, keepdims=True)

			# icm loss
			total_loss = tf.reduce_mean(0.2 * forward_loss + 0.8 * inverse_loss)
		
		#print('action_chosen')
		#print(action_chosen.numpy().tolist())
		print('pred_action')
		print(tf.math.reduce_sum(pred_action).numpy().tolist())
		grads = tape.gradient(total_loss, self.icm_model.trainable_weights) 
		#grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
		grads_vars = list(zip(grads, self.icm_model.trainable_weights))
		print('global grad')
		print(tf.linalg.global_norm(grads))

		self.curiosity_optimizer.apply_gradients(grads_vars)
		return tf.reduce_mean(forward_loss), tf.reduce_mean(inverse_loss), total_loss


	def _update_network(self, mb_total_returns, mb_states, mb_actions, mb_total_advantages, mb_old_policy):
		with tf.GradientTape() as tape:
			logits, values = self.global_model(mb_states) #shape=(t_steps*n_threads, n_actions), (t_steps*n_threads, 1)
			ext_values = values
			policy = tf.nn.softmax(logits) #shape=(n_step, n_actions)

			# Value loss
			ext_value_loss = tf.math.square(mb_total_returns - ext_values) #shape=(t_steps*n_threads, 1)
			value_loss = self.ext_val_coeff * ext_value_loss #+ self.int_val_coeff * int_value_loss

			# Entropy
			entropy = -tf.math.reduce_sum(policy * tf.math.log(policy), axis=-1, keepdims=True)  

			# Policy loss
			action_chosen = tf.one_hot(mb_actions, self.action_size, axis=-1, dtype=tf.float32)
			log_policy = tf.math.log(tf.math.reduce_sum(action_chosen * policy, axis=-1, keepdims=True))    

			log_old_policy = tf.math.log(tf.math.reduce_sum(action_chosen * mb_old_policy, axis=-1, keepdims=True))           
			ratio = tf.exp(log_policy - log_old_policy)
			clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1-self.ppo_clip, clip_value_max=1+self.ppo_clip)

			policy_loss = - tf.minimum(mb_total_advantages * clipped_ratio, mb_total_advantages * ratio)

			# Total loss
			total_loss = tf.reduce_mean(value_loss + policy_loss - self.entropy_coeff * entropy) 


		grads = tape.gradient(total_loss, self.global_model.trainable_weights) 
		#grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
		grads_vars = list(zip(grads, self.global_model.trainable_weights))

		self.optimizer.apply_gradients(grads_vars)


		return tf.reduce_mean(value_loss), tf.reduce_mean(policy_loss), tf.reduce_mean(entropy)


