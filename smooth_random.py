import numpy as np


def discrete_smooth_sampler(env, env_sample_chances = 0.1):
    '''
    A smoothed action sample for discrete environments. 
    env_sample_chances is either a float or a vector of action change changes. 
    If a vector each element represents a different chance of random sampling for the respective previous choice.

    e.g. if env_sample_chances[old_action] = 0.1 then the sampler will sample the environment in 10% of cases, 
    or will return old_action with 90% chance.
    
    '''

    change_chances = np.zeros(env.action_space.n, dtype = np.float64) + env_sample_chances
    last_action = [env.action_space.sample()]

    def pick_action(prev_action = None):
        if prev_action is None: prev_action = last_action[0]
        if np.random.random() > change_chances[prev_action]:
            new_action = last_action[0]
        else:
            new_action = env.action_space.sample()
        last_action[0] = new_action
        return new_action
    return pick_action


def continous_smooth_sampler(env, num_samples=10, temperature=0.1):
    """
    Samples multiple actions and picks one biased toward the previous one
    """
    last_action = [env.action_space.sample()] 

    def pick_action(): 
        # 1. Sample 10 random actions from the space
        candidate_actions = np.array([env.action_space.sample() for _ in range(num_samples)])

        # 2. Calculate Euclidean distances to the smoothing average
        # (Lower distance = closer to the 'trend')
        distances = np.linalg.norm(candidate_actions - last_action[0], axis=1) + 1e-8  # epsilon?

        # 3. Convert distances to weights
        # We use negative distance because Softmax picks the HIGHEST value.
        # Temperature controls how 'strict' the bias is (lower = more biased).
        weights = softmax(-distances / temperature)

        # 4. Pick one action based on the calculated probabilities
        chosen_idx = np.random.choice(num_samples, p=weights)
    
        last_action[0] = candidate_actions[chosen_idx]
        return last_action[0]

    return pick_action

def random_sampler(env):
    '''
    Just pick a "normal" random sample from the action space. 
    Same api as the smoothed ones
    '''
    def pick_action(prev_action = None): # prev_action just for compatibility with the other samplers
        return env.action_space.sample()
    return pick_action

# Helper Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def play_episode(env, action_sampler): 
    # collects a single episode rewards, states and actions
    # for the given sampler
    state, info = env.reset()
    rewards, states, actions = [], [], []
    while True:
        a = action_sampler()
        states.append(state)
        actions.append(a)
        state, rew, term, trunc, info = env.step(a)
        rewards.append(rew)
        if term or trunc: break 
    return rewards, states, actions
