import sys
sys.path.append('/home/mariosv/atari_rl/environments')


#from vanilla_ac import ActorCriticAgent
from curiosity_ppo import CuriosityActorCriticAgent
from wrapper import AtariPreprocessing

def main():
	TSTEPS = 128
	N_THREADS = 8 #8
	LEARNING_RATE = 0.0001 #this works 0.0001
	N_OPT_EPOCHS = 4
	N_MINIBATCHES_PER_EPOCH = 4
	EPOCH = 1000000
	N_EVALUATION_EP = 30
	N_STEPS = 10000000 #######################
	FRAME_STACK = 4
	MAX_EPISODE_STEPS = 4500


	GAME_NAME = 'PongDeterministic-v0'
	#GAME_NAME = 'MsPacmanDeterministic-v0'
	#GAME_NAME = 'MontezumaRevengeDeterministic-v4'
	#AGENT_NAME = 'A2C'
	#AGENT_NAME = 'PPO'
	#AGENT_NAME = 'RND01'
	AGENT_NAME = 'ICM005'
	SAVE_DIR = '/home/mariosv/atari_rl/results'


	'''agent = ActorCriticAgent(agent_name = AGENT_NAME, game_name=GAME_NAME, save_dir=SAVE_DIR, env_wrapper=AtariPreprocessing, frame_stack=FRAME_STACK, ppo=True)
	agent.train(t_steps=TSTEPS, n_opt_epochs=N_OPT_EPOCHS, n_minibatches_per_epoch=N_MINIBATCHES_PER_EPOCH, ppo_clip=0.2, entropy_coeff=0.01, n_steps=N_STEPS, max_episode_steps=MAX_EPISODE_STEPS, 
		reward_clip=True, n_threads=N_THREADS, learning_rate=LEARNING_RATE, eval_epoch_steps=EPOCH, n_eval_episodes=N_EVALUATION_EP)
	'''

	agent = CuriosityActorCriticAgent(agent_name = AGENT_NAME, game_name=GAME_NAME, save_dir=SAVE_DIR,  env_wrapper=AtariPreprocessing, frame_stack=FRAME_STACK, curiosity_module='icm')
	agent.train(t_steps=TSTEPS, n_opt_epochs=N_OPT_EPOCHS, n_minibatches_per_epoch=N_MINIBATCHES_PER_EPOCH, ppo_clip=0.2, entropy_coeff=0.01, n_steps=N_STEPS, max_episode_steps=MAX_EPISODE_STEPS, 
		reward_clip=True, n_threads=N_THREADS, learning_rate=LEARNING_RATE, n_obs_init_rollouts = 10, eval_epoch_steps=EPOCH, n_eval_episodes=N_EVALUATION_EP)

if __name__ == '__main__':
	main()
