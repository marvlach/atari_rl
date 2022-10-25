import sys
sys.path.append('/atari_rl/environments')

#from atari_rl.agents.deepq.deepq import DQNAgent
#from atari_rl.environments.wrapper import AtariPreprocessing
from deepq import DQNAgent
from wrapper import AtariPreprocessing

def main():
	BATCH_SIZE = 32                   # Batch size

	REPLAY_MEMORY_SIZE = 1000000      # Number of transitions stored in the replay memory

	FRAME_STACK = 4                   # agent history length

	TARGET_NETWORK_UPDATE_FREQ = 10000 # Number of steps between updating the target network. 
		                         


	TRAIN_INTERVAL = 4                # the number of actions selected by the agent between succesive 
		                          # q-network updates     

	LEARNING_RATE = 0.00025/4          #0.00025 for DQN, 0.00025/4 for all variants
		 
	MAX_EPISODE_LENGTH = 4500         # Equivalent of 5 minutes of gameplay at 60 frames per second
	EPOCH = 1000000                   # Number of frames the agent sees between evaluations
	N_EVALUATION_EP = 30              # Number episode for one evaluation
		     


	REPLAY_MEMORY_START_SIZE = 50000  # Number of completely random actions, 
		                          # before the agent starts learning
	N_STEPS = 25000000                 # Total number of steps the agent takes
	 


	#GAME_NAME = 'PongDeterministic-v0'
	GAME_NAME = 'MsPacmanDeterministic-v0'
	AGENT_NAME = 'PERdoubleDQN'
	SAVE_DIR = '/home/mariosv/atari_rl/results'
	dqnAgent = DQNAgent(agent_name = AGENT_NAME, game_name=GAME_NAME, save_dir=SAVE_DIR, env_wrapper=AtariPreprocessing, frame_stack=FRAME_STACK, double=True, duel=False)
	dqnAgent.train(n_steps=N_STEPS, reward_clip=True, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, 
		      target_network_update_freq=TARGET_NETWORK_UPDATE_FREQ, train_interval=TRAIN_INTERVAL, 
		      eval_epoch_steps=EPOCH, n_eval_episodes=N_EVALUATION_EP, max_episode_steps=MAX_EPISODE_LENGTH, 
		      replay_memory_size=REPLAY_MEMORY_SIZE, replay_memory_warmup_steps=REPLAY_MEMORY_START_SIZE,
		      init_eps=1.0, final_eps=0.1, test_eps=0.05, exploration_steps=1000000,
		      prioritized_replay=True, prioritized_replay_alpha=0.6, 
		      prioritized_replay_eps=1e-3, prioritized_replay_beta0=0.4) 


if __name__ == '__main__':
	main()
