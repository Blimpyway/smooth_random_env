import gymnasium as gym 
import numpy as np
from smooth_random import play_episode, random_sampler, continous_smooth_sampler, discrete_smooth_sampler 
from time import time, sleep

SHOW_ONCE = False  # Set true for a final visual rendering for the last env and sampler

def sampler_demo(env, pick_action, sampler_description, n_episodes = 20):
    print(f"\nRun {n_episodes} episodes of {sampler_description} for {env.spec.id}...", end = '', flush = True)
    total_rewards = []
    t = time()
    for i in range(n_episodes):
        rewards, states, actions = play_episode(env, pick_action); 
        total_rewards.append(sum(rewards))
    dt = time() - t
    print(f"done in {dt:.3f} seconds")
    print(f"Episode reward mean:{np.mean(total_rewards):.2f}, maximum: {max(total_rewards):.2f}, minimum:{min(total_rewards):.2f} \n")

env = gym.make('MountainCarContinuous-v0')

pick_action = random_sampler(env) 
sampler_demo(env, pick_action, "standard random sample")

pick_action = continous_smooth_sampler(env) 
sampler_demo(env, pick_action, "smoothed continous sample") 

env = gym.make("MountainCar-v0", render_mode = None) 

pick_action = random_sampler(env) 
sampler_demo(env, pick_action, "standard discrete random sample", n_episodes = 200)

pick_action = discrete_smooth_sampler(env, env_sample_chances = np.array([0.08, 0.6, 0.1])) 
sampler_demo(env, pick_action,  "smoothed discrete sampler", n_episodes = 200) # need more episodes for discrete mountain car

if SHOW_ONCE:
    env = gym.make("MountainCar-v0", render_mode = 'human') 
    play_episode(env, pick_action)
    sleep(3)  # keep the pygame window for few seconds
    env.close()
