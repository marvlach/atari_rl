import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import cv2
import numpy as np

class AtariPreprocessing(gym.Wrapper):
    
	def __init__(self, env, noop_max=10, frame_skip=1, xstart=0, xend=210, ystart=0, yend = 160,
				screen_size=84, terminal_on_life_loss=False, grayscale_obs=True,
				scale_obs=False, clip_reward=False):
		super().__init__(env)
		assert frame_skip > 0
		assert screen_size > 0
		assert noop_max >= 0
		'''if frame_skip > 1:
		assert 'NoFrameskip' in env.spec.id, 'disable frame-skipping in the original env. for more than one' \
		 ' frame-skip as it will be done by the wrapper'
		 '''
		self.noop_max = noop_max
		assert env.unwrapped.get_action_meanings()[0] == 'NOOP' 


		self.frame_skip = frame_skip
		self.screen_size = screen_size
		self.terminal_on_life_loss = terminal_on_life_loss
		self.grayscale_obs = grayscale_obs
		self.scale_obs = scale_obs
		self.clip_reward = clip_reward
		self.xstart = xstart
		self.xend = xend
		self.ystart = ystart
		self.yend = yend

		# buffer of most recent two observations for max pooling
		self.obs_buffer = np.empty(env.observation_space.shape[:2], dtype=np.uint8)
		self.ale = env.unwrapped.ale
		self.lives = 0
		self.game_over = False

		_low, _high, _obs_dtype = (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
		if grayscale_obs:
			self.observation_space = Box(low=_low, high=_high, shape=(screen_size, screen_size), dtype=_obs_dtype)
		else:
			self.observation_space = Box(low=_low, high=_high, shape=(screen_size, screen_size, 3), dtype=_obs_dtype)



	def step(self, action):
		R = 0.0 


		_, reward, done, _, info = self.env.step(action)
		if self.clip_reward:
			reward = np.sign(reward)

		R += reward
		self.game_over = done

		if self.terminal_on_life_loss:
			new_lives = self.ale.lives()
			done = done or new_lives < self.lives and new_lives > 0
			self.lives = new_lives


		if self.grayscale_obs:
			self.ale.getScreenGrayscale(self.obs_buffer)
		else:
			self.ale.getScreenRGB2(self.obs_buffer)

		return self._get_obs(), R, done, self.game_over, info

	def reset(self, **kwargs):
		# NoopReset
		self.env.reset(**kwargs)
		noops = self.unwrapped.np_random.integers(1, self.noop_max + 1) if self.noop_max > 0 else 0
		for _ in range(noops):
			_, _, done, _, _ = self.env.step(0) ############### 1 for breakout
			if done:
				self.env.reset(**kwargs)

		self.lives = self.ale.lives()
		if self.grayscale_obs:
			self.ale.getScreenGrayscale(self.obs_buffer)
		else:
			self.ale.getScreenRGB2(self.obs_buffer)
		#self.obs_buffer[1].fill(0)
		return self._get_obs()

	def _get_obs(self):

		'''if self.frame_skip > 1:  # more efficient in-place pooling
		np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])'''

		obs = self.obs_buffer[self.xstart:self.xend, self.ystart:self.yend]
		obs = cv2.resize(obs, (self.screen_size, self.screen_size), interpolation=cv2.INTER_NEAREST)

		if self.scale_obs:
			obs = np.asarray(obs, dtype=np.float32) / 255.0
		else:
			obs = np.asarray(obs, dtype=np.uint8)
		return obs
