# authors: anonymous

import numpy as np
from utils import choice

class Chain:
    def __init__(self, nb_states, stochasticity=1, proba_based=False, noise=0, gamma=0.99, value_ratio=0.5):
        # by default, the optimal value ratio between action 0 and action 1 is the one specified asan argument
        V_max = (stochasticity*gamma/(1-(1-stochasticity)*gamma))**(nb_states -2)
        proba = (1-gamma) * value_ratio * V_max / (1-value_ratio*gamma*V_max)
        reward_0 = value_ratio * V_max
        self.nb_states = nb_states
        self.nb_actions = 2
        self.proba = proba
        self.reward_0 = reward_0
        self.transition_function = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        self.is_done = False
        self.initial_state = 0
        self.current_state = self.initial_state
        self.final_state = nb_states - 1
        self.proba_based = proba_based
        self.stochasticity = stochasticity
        self._generate_transition_function()
        self._generate_reward_function()
        self.noise = noise

    def _generate_transition_function(self):
        for id_state in range(self.nb_states-1):
            if self.proba_based:
                self.transition_function[id_state, 0, id_state] = 1 - self.proba
                self.transition_function[id_state, 0, self.final_state] = self.proba
            else:
                self.transition_function[id_state, 0, self.final_state] = 1
            self.transition_function[id_state, 1, id_state + 1] = self.stochasticity
            self.transition_function[id_state, 1, id_state] = 1-self.stochasticity

    def reset(self):
        self.current_state = self.initial_state
        return int(self.current_state)

    def sample_action(self):
        return np.random.randint(self.nb_actions)

    def _get_reward(self, state, action, next_state):
        r = self.reward_function[state, next_state] + (2*np.random.rand()-1)*self.noise 
        return r

    def step(self, state, action):
        next_state = choice(self.nb_states, self.transition_function[int(state), action, :].squeeze())
        reward = self._get_reward(state, action, next_state)
        is_done = (next_state == self.final_state)
        return reward, next_state, is_done

    # Reward matrix
    def _generate_reward_function(self):
        self.reward_function = np.zeros((self.nb_states, self.nb_states))
        if self.proba_based:
            r = 1
        else:
            r = self.reward_0
        for s in range(self.nb_states):
            self.reward_function[s, self.final_state] = r
        self.reward_function[self.final_state - 1, self.final_state] = 1
        return self.reward_function

    def start_state(self):
        return self.initial_state

    def generate_random_policy(self):
        # We use log to have a policy that is further from uniform.
        x = np.log(np.random.rand(self.nb_states,self.nb_actions))
        pi = x/x.sum(axis=1).reshape(self.nb_states,1)
        return pi
