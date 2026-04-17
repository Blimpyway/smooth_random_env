# smooth_random_env
Smoothed random action sampling for gymnasium-style RL environments

Various algorithms in RL either make use of an occasional "explore" random action or collect initial random episodes to bootstrap the training. 

However, a general issue with random sampling - specialy for delta-time physics simulating environments - is that the  
actions average over a median point within the action space. 

This makes the agent's "random" trajectory wiggling close to one applying the average over the action space.
e.g. in CarRacing it would  incoherently slam random steering, throtle and brakes, resulting in a short, low reward trajectory or in MountainCar a random action doesn't move the cart too far before the episode ends.

Files:

smooth_random.py 
----------------

provides a few tools for smoothing the sampled action for both continuous and discrete action environments 
and to collect environment training data regardless of observation returned by environment. 

They action samplers are: 
discrete_smooth_sampler()  - for discrete action environments 

continous_smooth_sampler() - for continuous action environments

random_sampler()           - sampler api semantics for environments own .action_space.sample() 


All above samplers return an action_pick function which can be used in play_episode() below: 
the action_pick() returns a new action "closer" to the one from a previous call. 

play_episode(env, action_pick) - play a new episode for the specified environment and action picker



demo_mountain_car.py 
--------------------

Showcases the use of smooth_random.py above for both continous and discrete versions of the MountainCar environment 
(MountainCarContinuous-v0 and MountainCar-v0) 


within this folder run:

python demo_mountain_car.py 

Here-s a copy&paste of the printed result: 


--------------------

Run 20 episodes of standard random sample for MountainCarContinuous-v0...done in 1.271 seconds
Episode reward mean:-33.46, maximum: -30.93, minimum:-34.45    
# All episodes equally bad

Run 20 episodes of smoothed continous sample for MountainCarContinuous-v0...done in 7.137 seconds
Episode reward mean:65.04, maximum: 95.93, minimum:-31.21      
# A lot better average and maximum reward episode

Run 200 episodes of standard discrete random sample for MountainCar-v0...done in 1.251 seconds
Episode reward mean:-200.00, maximum: -200.00, minimum:-200.00 
# Discrete action version very hard to solve randomly or to nudge constant flat results despite 10x more attempts

Run 200 episodes of smoothed discrete sampler for MountainCar-v0...done in 1.083 seconds
Episode reward mean:-192.10, maximum: -92.00, minimum:-200.00   
# This provides several episodes with better results and ocassionally pretty good ones

-----------------------------

Check the source for detail usage

------------

Requirements: gymnasium, numpy

------------

License: Apache 2.0

