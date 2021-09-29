# authors: anonymous

import numpy as np
import utils
from utils import prt
from utils import choice

class Garnets:
    def __init__(self, nb_states, nb_actions, nb_next_state_transition, env_type, self_transitions=0):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.nb_next_state_transition = nb_next_state_transition
        self.transition_function = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        self.is_done = False
        self.initial_state = 0
        self.self_transitions = self_transitions
        self.current_state = self.initial_state
        self.final_state = nb_states - 1
        self._generate_transition_function()
        self._generate_reward_function()
        self.env_type = env_type

    def _generate_transition_function(self):
        for id_state in range(self.nb_states):
            for id_action in range(self.self_transitions):
                self_transition_prob = np.random.uniform(0.5, 1)
                partition = np.sort(np.random.uniform(0, 1-self_transition_prob, self.nb_next_state_transition - 2))
                partition = np.concatenate(([0], partition, [1-self_transition_prob]))
                probabilities = np.ediff1d(partition)
                choice_state = np.random.choice(self.nb_states, self.nb_next_state_transition - 1, replace=False)
                self.transition_function[id_state, id_action, choice_state] = probabilities
                self.transition_function[id_state, id_action, id_state] += self_transition_prob

            for id_action in range(self.self_transitions, self.nb_actions):
                partition = np.sort(np.random.uniform(0, 1, self.nb_next_state_transition - 1))
                partition = np.concatenate(([0], partition, [1]))
                probabilities = np.ediff1d(partition)
                choice_state = np.random.choice(self.nb_states, self.nb_next_state_transition, replace=False)
                self.transition_function[id_state, id_action, choice_state] = probabilities

    # Reward matrix
    def _generate_reward_function(self):
        self.reward_function = np.zeros((self.nb_states, self.nb_states))
        for s in range(self.nb_states):
            self.reward_function[s, self.final_state] = 1
        return self.reward_function

    def reset(self):
        self.current_state = self.initial_state
        return int(self.current_state)

    def sample_action(self):
        return int(choice(self.nb_actions, 1))

    def _get_reward(self, state, action, next_state):
        if next_state == self.final_state:
            return 1
        else:
            return 0

    def step(self, state, action):
        if self.transition_function[state, action, :].squeeze().sum() != 1:
            prt(self.transition_function[state, action, :].squeeze())
            prt(state)
            prt(self.final_state)
            prt(state==self.final_state)
        next_state = choice(self.nb_states, self.transition_function[state, action, :].squeeze())
        reward = self._get_reward(state, action, next_state)
        is_done = (next_state == self.final_state)
        return reward, next_state, is_done

    # Reward matrix
    def compute_reward(self):
        R = np.zeros((self.nb_states, self.nb_states))
        for s in range(self.nb_states):
            R[s, self.final_state] = 1
        return R

    # Transition function matrix
    def compute_transition_function(self):
        t = self.transition_function.copy()
        t[self.final_state, :, :] = 0
        return t

    def start_state(self):
        return self.initial_state

    def generate_random_policy(self):
        # We use log to have a policy that is further from uniform.
        x = np.log(np.random.rand(self.nb_states,self.nb_actions))
        pi = x/x.sum(axis=1).reshape(self.nb_states,1)

        return pi

    def generate_baseline_policy(self, gamma, softmax_target_perf_ratio=0.75,
                                 baseline_target_perf_ratio=0.5, softmax_reduction_factor=0.9,
                                 perturbation_reduction_factor=0.9, farthest=True):
        if softmax_target_perf_ratio < baseline_target_perf_ratio:
            softmax_target_perf_ratio = baseline_target_perf_ratio

        farther_state, pi_star_perf, q_star, pi_rand_perf = self._find_farther_state(gamma, farthest)
        p, r = self._set_temporary_final_state(farther_state)
        self.transition_function = p.copy()
        self._generate_reward_function()
        r_reshaped = utils.get_reward_model(p, r)
        if softmax_target_perf_ratio < 0:
            return None, None, pi_star_perf, None, pi_rand_perf, farther_state

        softmax_target_perf = softmax_target_perf_ratio * (pi_star_perf - pi_rand_perf) \
                              + pi_rand_perf
        pi, _, _ = self._generate_softmax_policy(q_star, p, r_reshaped,
                                                 softmax_target_perf, 
                                                 softmax_reduction_factor, gamma)

        baseline_target_perf = baseline_target_perf_ratio * (pi_star_perf - pi_rand_perf) \
                              + pi_rand_perf
        pi, v, q = self._perturb_policy(pi, q_star, p, r_reshaped, baseline_target_perf,
                             perturbation_reduction_factor, gamma) 

        return pi, q, pi_star_perf, v[0], pi_rand_perf, farther_state
        
    def _perturb_policy(self, pi, q_star, p, r_reshaped, baseline_target_perf,
                        reduction_factor, gamma):
        v = np.ones(1) 
        while v[0] > baseline_target_perf:
            x = np.random.randint(self.nb_states)
            pi[x, np.argmax(q_star[x,:])] *= reduction_factor
            pi[x, :] /= np.sum(pi[x,:])
            v, q = utils.policy_evaluation_exact(pi, r_reshaped, p, gamma)

        avg_time_to_goal = np.log(v[0])/np.log(gamma)
        prt("Perturbed policy performance : " + str(v[0]))
        prt("Perturbed policy average time to goal: " + str(avg_time_to_goal))
        return pi, v, q

    def _generate_softmax_policy(self, q_star, p, r_reshaped, softmax_target_perf,
                                 reduction_factor, gamma):
        temp = 2*10**6 # Actually starts exploring for half its value.
        v = np.ones(1)
        while v[0] > softmax_target_perf:
            temp *= reduction_factor
            pi = utils.softmax(q_star, temp)
            v, q = utils.policy_evaluation_exact(pi, r_reshaped, p, gamma)

        avg_time_to_goal = np.log(v[0])/np.log(gamma)
        prt("Softmax performance : " + str(v[0]))
        prt("Softmax temperature : " + str(temp))
        prt("Softmax average time to goal: " + str(avg_time_to_goal))
        return pi, v, q

    def _set_temporary_final_state(self, final_state):
        self.final_state = final_state
        p = self.compute_transition_function()
        r = self.compute_reward()
        return p, r

    def _find_farther_state(self, gamma, farthest=True):
        argmin = -1
        min_value = 1
        rand_value = 0
        best_q_star = 0
        rand_pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        if farthest:
            final_states = range(1, self.nb_states)
        else:
            final_states = [self.nb_states-1]
        for final_state in final_states:
            p, r = self._set_temporary_final_state(final_state)
            r_reshaped = utils.get_reward_model(p, r)

            rl = utils.policyiteration(gamma, self.nb_states, self.nb_actions, p, r_reshaped)
            rl.fit()
            v_star, q_star = utils.policy_evaluation_exact(rl.pi, r_reshaped, p, gamma)
            v_rand, q_rand = utils.policy_evaluation_exact(rand_pi, r_reshaped, p, gamma)

            perf_star = v_star[0]
            perf_rand = v_rand[0]

            if perf_star < min_value and perf_star > gamma**50:
                min_value = perf_star
                argmin = final_state
                rand_value = perf_rand
                best_q_star = q_star.copy()

        # avg_time_to_goal = np.log(min_value)/np.log(gamma)
        # avg_time_to_goal_rand = np.log(rand_value)/np.log(gamma)
        # prt("Optimal performance : " + str(min_value))
        # prt("Optimal average time to goal: " + str(avg_time_to_goal))
        # prt("Random policy performance : " + str(rand_value))
        # prt("Random policy average time to goal: " + str(avg_time_to_goal_rand))

        return argmin, min_value, best_q_star, rand_value
