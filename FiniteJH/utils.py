# authors: anonymous

import numpy as np
import time


# 25x faster than np.random.choice ...
def choice(options, probs):
    if isinstance(options, int):
        options = range(options)
    x = np.random.rand()
    cum = 0
    for i,p in enumerate(probs):
        cum += p
        if x < cum:
            return options[i]
    return options[-1]

# Sectect action based on the the action-state function with a softmax strategy
def softmax_action(Q, s):
    proba=np.exp(Q[s, :])/np.exp(Q[s, :]).sum()
    nb_actions = Q.shape[1]
    return choice(nb_actions, proba)


# Prints with a time stamp
def prt(s):
    format1 = ';'.join([str(0), str(30), str(41)])
    format2 = ';'.join([str(0), str(31), str(40)])
    s1 = '\x1b[%sm %s \x1b[0m' % (format1, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    s2 = '\x1b[%sm %s \x1b[0m' % (format2, s)
    print(s1 + '  '+ s2)


# The reward function is defined on SxS, but we need it on SxA.
# This function makes the transformation based on the transition function P.
def get_reward_model(P, R):
    return np.einsum('ijk,ik->ij', P, R)



def policy_evaluation_exact(pi, r, p, gamma):
    """
    Evaluate policy by taking the inverse
    Args:
      pi: policy, array of shape |S| x |A|
      r: the true rewards, array of shape |S| x |A|
      p: the true state transition probabilities, array of shape |S| x |A| x |S|
    Return:
      v: 1D array with updated state values
      q: 2D array with updated state-action values
    """
    # Rewards according to policy: Hadamard product and row-wise sum
    r_pi = np.einsum('ij,ij->i', pi, r)

    # Policy-weighted transitions:
    # multiply p by pi by broadcasting pi, then sum second axis
    # result is an array of shape |S| x |S|
    p_pi = np.einsum('ijk, ij->ik', p, pi)
    v = np.dot(np.linalg.inv((np.eye(p_pi.shape[0]) - gamma * p_pi)), r_pi)
    return v, r + gamma*np.einsum('i, jki->jk', v, p)

   
# Computes the softmax from a value function and a temperature. 
def softmax(q, temp=1):
    exp = np.exp(temp*(q - np.max(q, axis=1)[:,None]))
    pi = exp / np.sum(exp, axis=1)[:,None]
    return pi

# sampling from a state_action buffer: 
def sample_from(arr):
    s = choice(np.arange(arr.shape[0]), arr.sum(axis=1)/arr.sum())
    a = choice(np.arange(arr.shape[1]), arr[s]/arr[s].sum())
    return s,a

class policyiteration():
    # gamma is the discount factor,
    # nb_states is the number of states in the MDP,
    # nb_actions is the number of actions in the MDP,
    # model is the transition model,
    # reward is the reward model,
    # max_nb_it is the maximal number of policy improvement
    def __init__(self, gamma, nb_states, nb_actions, model, reward, max_nb_it=99999):
        self.gamma = gamma
        self.nb_actions = nb_actions
        self.nb_states = nb_states
        self.P = model
        self.R = reward.reshape(self.nb_states * self.nb_actions)
        self.max_nb_it = max_nb_it

    # starts a new episode (during the policy exploitation)
    def new_episode(self):
        self.has_bootstrapped = False

    # trains the policy
    def fit(self):
        pi = np.ones((self.nb_states, self.nb_actions))/self.nb_actions
        q = np.zeros((self.nb_states, self.nb_actions))
        old_q = np.ones((self.nb_states, self.nb_actions))
        nb_sa = self.nb_states * self.nb_actions
        nb_it = 0
        old_pi = None
        while np.linalg.norm(q - old_q) > 0.000000001 and nb_it < self.max_nb_it:
            old_q = q.copy()
            M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', self.P, pi).reshape(nb_sa, nb_sa)
            q = np.dot(np.linalg.inv(M), self.R).reshape(self.nb_states, self.nb_actions)
            pi = self.update_pi(q, old_pi)
            old_pi = pi
            nb_it += 1
        self.pi = pi
        self.q = q

    # does the policy improvement inside the policy iteration loop
    def update_pi(self, q, old_pi=None):
        pi = np.zeros(q.shape)
        for s in range(self.nb_states):
            pi[s, np.argmax(q[s, :])] = 1
        return pi

    # implements the trained policy
    def predict(self, state):
        return choice(self.nb_actions, self.pi[state])
