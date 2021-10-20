import sys
sys.path.append('/home/mariosv/atari_rl')
import os
import tensorflow as tf
import numpy as np
import gym
import csv
from gym.wrappers import Monitor
from memory import ExperienceReplayMemory, PrioritizedExperienceReplayMemory
from models import DQN, duelDQN
from policy import EpsGreedyPolicy
from util import LinearSchedule




class DQNAgent(object):
	def __init__(self, agent_name, game_name, save_dir, env_wrapper, frame_stack, double, duel):
		self.agent_name = agent_name
		self.game_name = game_name
		self.save_dir = save_dir
		self.env_wrapper = env_wrapper
		self.frame_stack = frame_stack
		self.double = double
		self.duel = duel
		if self.env_wrapper is not None: # for atari
			self.env = self.env_wrapper(gym.make(self.game_name))
			self.test_env = self.env_wrapper(Monitor(gym.make(self.game_name), 
								os.path.join(self.save_dir, self.agent_name, self.game_name, 'video'),  
								force=True, video_callable=lambda episode_id: episode_id % 10 == 0))
		else:
			self.env = gym.make(self.game_name)
			self.test_env = Monitor(gym.make(self.game_name), 
						os.path.join(self.save_dir, self.agent_name, self.game_name, 'video'),
						force=True, video_callable=lambda episode_id: episode_id % 10 == 0)


		self.obs_shape = self.env.observation_space.shape

		# assert that observations are images
		assert len(self.obs_shape) == 2, "Observations must be grayscale images"

		if self.frame_stack > 1:#Atari (x, x, frame_stack)
			self.state_shape = self.obs_shape + (self.frame_stack, )
		else:#
			self.state_shape = self.obs_shape

		self.action_size = self.env.action_space.n


		self.gamma = 0.99
		
		if self.duel:
			self.target_dqn = duelDQN(gym.make(self.game_name).action_space.n) # non trainable model
			self.target_dqn(tf.convert_to_tensor(np.random.random((32,) + self.env_wrapper(gym.make(self.game_name)).observation_space.shape + (self.frame_stack,)), dtype=tf.float32))

			self.main_dqn = duelDQN(gym.make(self.game_name).action_space.n)  # trainable model
			self.main_dqn(tf.convert_to_tensor(np.random.random((32,) + self.env_wrapper(gym.make(self.game_name)).observation_space.shape + (self.frame_stack,)), dtype=tf.float32))

			self.test_model = duelDQN(gym.make(self.game_name).action_space.n)  # model used for evaluation
			self.test_model(tf.convert_to_tensor(np.random.random((32,) + self.env_wrapper(gym.make(self.game_name)).observation_space.shape + (self.frame_stack,)), dtype=tf.float32))

		else:
			self.target_dqn = DQN(gym.make(self.game_name).action_space.n)  # non trainable model
			self.target_dqn(tf.convert_to_tensor(np.random.random((32,) + self.env_wrapper(gym.make(self.game_name)).observation_space.shape + (self.frame_stack,)), dtype=tf.float32))

			self.main_dqn = DQN(gym.make(self.game_name).action_space.n)  # trainable model
			self.main_dqn(tf.convert_to_tensor(np.random.random((32,) + self.env_wrapper(gym.make(self.game_name)).observation_space.shape + (self.frame_stack,)), dtype=tf.float32))

			self.test_model = DQN(gym.make(self.game_name).action_space.n)  # model used for evaluation
			self.test_model(tf.convert_to_tensor(np.random.random((32,) + self.env_wrapper(gym.make(self.game_name)).observation_space.shape + (self.frame_stack,)), dtype=tf.float32))


		#self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1.5e-04)
		#self.loss_function = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
		#self.loss_function = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


	def evaluate(self, best_avg100_reward, eps, n_eval_episodes):
		# set test model, global model keeps updating by threads in parallel
		self.test_model.set_weights(self.main_dqn.get_weights())
		print('Evaluating for {} episodes...'.format(n_eval_episodes))
		test_steps = []
		test_rewards = []



		for ep in range(n_eval_episodes):
			ep_rew, ep_step = self.play(self.test_env, eps)
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

	def play(self, env, eps, model_dir = None):

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
			q_values = model(tf.convert_to_tensor(current_state[None, :], dtype=tf.float32))
			action = self.policy.get_action(eps, q_values.numpy()[0])
			new_obs, reward, done, _, _ = env.step(action)

			if self.frame_stack > 1:
				new_state = np.append(current_state[:, :, 1:], np.expand_dims(new_obs, axis=2), axis=2)
			else:
				new_state = new_obs


			ep_reward += reward
			ep_steps += 1
			current_state = new_state

		return ep_reward, ep_steps



	def train(self, n_steps, reward_clip, learning_rate, batch_size, 
			target_network_update_freq, train_interval, 
			eval_epoch_steps, n_eval_episodes, max_episode_steps, 
			replay_memory_size, replay_memory_warmup_steps,
			init_eps, final_eps, test_eps, exploration_steps,
			prioritized_replay, prioritized_replay_alpha, 
			prioritized_replay_eps, prioritized_replay_beta0):



		self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1.5e-04)

		self.loss_function = tf.keras.losses.Huber()

		self.policy = EpsGreedyPolicy(action_size=self.action_size)
		self.EpsSchedule = LinearSchedule(initial_value=init_eps, final_value=final_eps, 
		total_steps=exploration_steps, warmup_steps=replay_memory_warmup_steps)
		if prioritized_replay:
			self.BetaSchedule = LinearSchedule(initial_value=prioritized_replay_beta0, final_value=1, 
			total_steps=50e6, warmup_steps=0)
			self.memory = PrioritizedExperienceReplayMemory(size=replay_memory_size, obs_stack=self.frame_stack, alpha=prioritized_replay_alpha)
		else:
			self.memory = ExperienceReplayMemory(size=replay_memory_size, obs_stack=self.frame_stack)

		global_steps = 0
		best_avg100_reward = -21 ####################
		reward_list = []
		steps_list = []

		self._update_target_network() 




		while global_steps < n_steps:
			epoch_steps = 0

			while epoch_steps < eval_epoch_steps:
				# reset env, get first frame and repeat it to form initial state
				current_obs = self.env.reset()
				if self.frame_stack > 1: #atari equiv. to if current_obs.ndim == 22
					current_state = np.repeat(np.expand_dims(current_obs, axis=2), self.frame_stack, axis = 2)   
				else:
					current_state = current_obs

				ep_reward = 0
				ep_steps = 0
				done_ep = False
				while ep_steps < max_episode_steps and not done_ep:

					# get logits of state
					logits = self.main_dqn(tf.convert_to_tensor(current_state[None, :], dtype=tf.float32))

					# get epsilon according to global_steps
					eps = self.EpsSchedule.get_eps(global_steps)
					action = self.policy.get_action(eps, logits.numpy()[0])

					# take action
					new_obs, reward, done_life, done_ep, _ = self.env.step(action)

					#stack new obs to current state to form new_state
					if self.frame_stack > 1:
						new_state = np.append(current_state[:, :, 1:], np.expand_dims(new_obs, axis=2), axis=2)
					else:
						new_state = new_obs


					ep_reward += reward
					ep_steps += 1
					clipped_reward = reward
					if reward_clip:
						clipped_reward = np.sign(reward)


					global_steps += 1
					epoch_steps += 1
					# add transition to replay buffer# dont expand for cartpole
					self.memory.store(obs=np.expand_dims(new_obs, axis=new_obs.ndim), action=action, reward=clipped_reward, done=done_life, print_proc=True if global_steps % 100000 == 0 else False) 
					# perform one update every train_interval steps
					if global_steps % train_interval == 0 and global_steps > replay_memory_warmup_steps:
						if prioritized_replay:
							mb_states, mb_actions, mb_rewards, mb_new_states, mb_dones, mb_is_weights, mb_indices = self.memory.sample_minibatch(
								batch_size, beta=self.BetaSchedule.get_eps(global_steps), print_proc=True if global_steps % 100000 == 0 else False)  
						else:
							mb_states, mb_actions, mb_rewards, mb_new_states, mb_dones = self.memory.sample_minibatch(batch_size)
							mb_is_weights = [1.0]  

						q_targets = self._calculate_q_targets(np.concatenate(mb_new_states, axis=0), np.array(mb_rewards), np.array(mb_dones))
						#debugging memory
						for i in range(batch_size):
							assert np.array_equal(mb_states[i][:, :, :, 1:], mb_new_states[i][:, :, :, :-1]), 'mem state encode is bugged'
							#indices_hist[mb_indices[i]] += 1

						td_errors = self._update_Q_network(tf.convert_to_tensor(np.concatenate(mb_states, axis=0), dtype=tf.float32),
													tf.convert_to_tensor(np.array(mb_actions), dtype=tf.int32), 
													tf.convert_to_tensor(q_targets[:, None], dtype=tf.float32),
													tf.convert_to_tensor(np.array(mb_is_weights)[:, None], dtype=tf.float32))


						if prioritized_replay:
							mb_new_priorities = np.squeeze(np.abs(td_errors.numpy()) + prioritized_replay_eps).tolist()
							self.memory.update_priorities(mb_indices, mb_new_priorities, print_proc=True if global_steps % 10000 == 0 else False)


					# update target network    
					if global_steps % target_network_update_freq == 0 and global_steps > replay_memory_warmup_steps:
						self._update_target_network() 

					# new state
					current_state = new_state

				# end of episode
				episode_msg = 'global steps=' + str(global_steps)
				episode_msg += ', episode '+str(len(reward_list))
				episode_msg += ', reward= '+ str(ep_reward)
				episode_msg += ', steps=' + str(ep_steps)
				episode_msg += ', epsilon=' + str(eps)
				#episode_msg += ', beta=' + str(self.BetaSchedule.get_eps(global_steps))
				print(episode_msg)
				reward_list.append(ep_reward)
				steps_list.append(ep_steps)



			# end of epoch: evaluate for n_eval_episodes
			best_avg100_reward = self.evaluate(best_avg100_reward, test_eps, n_eval_episodes)



		#end of training
		# write results in csv
		with open(os.path.join(self.save_dir, self.agent_name, self.game_name, 'learning_curve.csv'), 'w', newline='') as file:
			writer = csv.writer(file)
			for i in range(len(reward_list)):
				writer.writerow([i, steps_list[i], reward_list[i]])

		# save final model weights
		self.main_dqn.save_weights(os.path.join(self.save_dir, self.agent_name, self.game_name, 'final_model.h5'))



	def _update_target_network(self):
		self.target_dqn.set_weights(self.main_dqn.get_weights())


	def _calculate_q_targets(self, new_states, rewards, terminal):
		if self.double:
			# The main network estimates the best action from next state(s'), while
			# The target network estimates the q-values of the next state(s') for that action
			next_best_action = np.argmax(self.main_dqn(tf.convert_to_tensor(new_states, dtype=tf.float32)).numpy(), axis=-1)
			target_q_values = self.target_dqn(tf.convert_to_tensor(new_states, dtype=tf.float32)).numpy()
			target_q = rewards + self.gamma * target_q_values[np.arange(target_q_values.shape[0]), next_best_action] * (1 - terminal)

		else:
			# The target network estimates the q-values of the next state(s') as well as
			# the best action a' from s' for every next_state of the minibatch(no double q)
			target_q_values = self.target_dqn(tf.convert_to_tensor(new_states, dtype=tf.float32)).numpy()
			target_q = rewards + self.gamma * np.amax(target_q_values, axis=-1) * (1 - terminal)

		return target_q


	@tf.function
	def _update_Q_network(self, states, actions, q_targets, importance_sample_weights): #, rewards, new_states, terminal_flags):

		action_taken = tf.one_hot(actions, self.action_size, dtype=tf.float32)

		with tf.GradientTape() as tape:
			q_values = self.main_dqn(states)

			q_value_a = tf.math.reduce_sum(tf.multiply(q_values, action_taken), axis=-1, keepdims=True)

			loss = self.loss_function(tf.stop_gradient(q_targets), q_value_a, importance_sample_weights)

		grads = tape.gradient(loss, self.main_dqn.trainable_weights)#loss
		self.optimizer.apply_gradients(zip(grads, self.main_dqn.trainable_weights))
		return q_targets - q_value_a



