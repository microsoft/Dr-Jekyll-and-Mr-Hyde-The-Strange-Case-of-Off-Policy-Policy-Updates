# authors: anonymous
import numpy as np
import proj_simplex
from utils import prt
from utils import softmax
from utils import get_reward_model
from utils import sample_from
from utils import choice


class GradientAscent():
    # gamma is the discount factor,
    # nb_states is the number of states in the MDP,
    # nb_actions is the number of actions in the MDP,
    # env is the environment
    # model is the transition model,
    # reward is the reward model,
    # max_nb_it is the maximal number of policy improvement (in exact setting) or the number of trajectories (in sample setting)
    # perf_stop is used in some experiments when you want to stop learning when some leel of performance is reached
    def __init__(self, gamma, nb_states, nb_actions, env, model, reward,
                 max_nb_it=1000, perf_stop=2):
        self.gamma = gamma
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.P = model
        self.R = get_reward_model(self.P, reward).reshape(self.nb_actions*self.nb_states)
        self.max_nb_it = max_nb_it
        self.env = env
        self.perf_stop = perf_stop

    # implements the trained policy
    def perform_action(self, state, pi):
        action = choice(range(len(pi[state])), pi[state])
        return action

    # Trains a policy on the exact gradients like in theory (q and d are assumed to be known)
        # discounting defines the type of algorithm
        # actor_stepsize is the learning rate of the actor
        # hyde_param is the parameter factor for the probability of selecting Hyde's policy (only used with discounting='jh')
        # alpha is the decaying power of the Hyde policy selection
        # lambada (cannot use lambda ;-)) is the parameter for entropic regularization
    def fit_exact(self, parametrization, discounting, actor_stepsize, hyde_param = 10, alpha=1, lambada=0):
        prt("exact " + parametrization + ' ' + discounting)
        
        nb_sa = self.nb_actions*self.nb_states

        ####### Initializations #######
        # policy parameters
        theta = np.zeros((self.nb_states, self.nb_actions))
        # main policy (also named Jekyll)
        pi = softmax(theta)
        # Number of updates until now
        nb_it = 0
        # Performance vector of the main policy
        perf_vect = []
        
        while nb_it < self.max_nb_it:
            # Computes the exact distribution of the main policy in the true MDP
            M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', self.P, pi).reshape(nb_sa, nb_sa)
            sa_0 = np.zeros(nb_sa)
            sa_0[:self.nb_actions] = pi[0]
            d_pi = np.linalg.inv(M)
 
            # ACTOR UPDATE
            if discounting == 'theory':
                # discounted distribution induced by the main policy
                d_0 = np.dot(sa_0, d_pi)
            if discounting == 'jh':
                # distribution induced by the main policy
                d_jekyll = np.dot(sa_0, d_pi) # distribution induite par la politique parametree
                d_jekyll /= d_jekyll.sum() # we normalise to measure exactly the jekyll/hyde ratio
                # uniform distribution (does not correspond to any policy)
                # we normalise to measure exactly the jekyll/hyde ratio, 
                # -1 because the terminal state is not a state where we take actions.
                pi_copy = pi.copy()
                pi_copy[self.env.final_state] = 0
                d_hyde = pi_copy.reshape(nb_sa)/(self.nb_states-1)
                # ratio of Hyde distribution selection
                ratio = min(1, hyde_param / (nb_it+1)**alpha)
                # ratio-weighting of the distribution
                d_0 = (1-ratio) * d_jekyll + ratio * d_hyde
            elif discounting == 'practice':
                # undiscounted distribution induced by the main policy
                M_gamma_less = np.eye(nb_sa) - np.einsum('ijk,kl->ijkl', self.P, pi).reshape(nb_sa, nb_sa)
                d_pi_gamma_less = np.linalg.inv(M_gamma_less)
                d_0 = np.dot(sa_0, d_pi_gamma_less)
            # DEPRECATED (won't be in the paper)
            # elif discounting == 'supercounted':
            #     M_cumul = np.eye(nb_sa) - (1/self.gamma) * np.einsum('ijk,kl->ijkl', self.P, pi).reshape(nb_sa, nb_sa)
            #     d_pi_cumul = np.linalg.inv(M_cumul)
            #     d_0 = np.dot(sa_0, d_pi_cumul)
            # DEPRECATED (won't be in the paper)
            # elif discounting == 'cumul': # gamma'=gamma
            #     d_0 = np.dot(sa_0, d_pi)
            #     d_0 = np.dot(d_0, d_pi)
            # DEPRECATED (won't be in the paper)
            # elif discounting == 'cumul2': # gamma'=1
            #     M_gamma_less = np.eye(nb_sa) - np.einsum('ijk,kl->ijkl', self.P, pi).reshape(nb_sa, nb_sa)
            #     d_pi_gamma_less = np.linalg.inv(M_gamma_less)
            #     d_0 = np.dot(sa_0, d_pi_gamma_less)
            #     d_0 = np.dot(d_0, d_pi)

            # Rescale updates on the same weight
            d_0 /= d_0.sum()

            # True value
            q = np.dot(d_pi, self.R)
            # True performance 
            perf = np.dot(q,sa_0)
            perf_vect.append(perf)

            # Stops if a stopping criterion on performance has been met
            if perf > self.perf_stop:
                return nb_it

            if parametrization == 'softmax':
                # True advantage
                adv = (q.reshape(self.nb_states, self.nb_actions)-lambada*np.log(pi)-np.sum(q.reshape(self.nb_states, self.nb_actions)*pi, axis=1).reshape(self.nb_states, 1)).reshape(nb_sa)
                # Gradient (no pi because d_0(s,a) = d_0(s)*pi(a|s))
                grad = (adv*d_0).reshape(self.nb_states, self.nb_actions)
                # Update of parameters
                theta += actor_stepsize*grad
                # For numerical stability, rescale the thetas.
                theta -= theta.max(axis=1).reshape(self.nb_states,1)
                # Update the main policy
                pi = softmax(theta)
            elif parametrization == 'direct':
                # Gradient (instead of dividing d_0(s,a) by pi(s,a) which may cause numerical instability, we use d_0(s) = sum_a d_0(s,a))
                grad = q.reshape(self.nb_states, self.nb_actions)*d_0.reshape(self.nb_states, self.nb_actions).sum(axis=1).reshape(self.nb_states,1)
                pi = proj_simplex.projection_simplex_sort(pi+actor_stepsize*grad)

            # Update counters
            nb_it += 1
           
        # Returns the maximal value if a stopping criterion on performance has been met 
        if self.perf_stop < 1:
            return self.max_nb_it
        # Otherwise returns the recording of the performance across time
        return perf_vect


    # trains the policy from samples with softmax parametrization
        # discounting defines the type of algorithm
        # critic_type is the type of algorithm used for critic. I recommend to use 'mle': less parameters.
        # critic_stepsize is the learning rate of the critic (not used with critic_type='mle')
        # init_q is the value at which the critic is initialize (not used with critic_type='mle'). When negative, the initialization is uniformly random in [0,1] iid in each state-action pair.
        # actor_stepsize is the learning rate of the actor
        # critic_UCB_stepsize is the learning rate of the UCB critic (not used with critic_type='mle')
        # hyde_param is the parameter factor for the probability of selecting Hyde's policy (only used with discounting='jh')
        # alpha is the decaying power of the Hyde policy selection
        # lambada (cannot use lambda ;-)) is the parameter for entropic regularization
    def fit_samples(self, parametrization, discounting, critic_type, actor_stepsize, critic_stepsize=0.1, init_q=1,
                             hyde_param=10, alpha=1, lambada=0, critic_UCB_stepsize=0, minibatch_size=1, offpol_param=0,
                             offpol_alpha=0):
        prt("samples " + parametrization + ' ' + discounting)
        
        nb_sa = self.nb_actions*self.nb_states

        ####### Initializations #######
        # policy parameters
        theta = np.zeros((self.nb_states, self.nb_actions))
        # main policy (also named Jekyll)
        pi = softmax(theta)
        # value estimate of main policy
        if init_q < 0:
            q = np.random.rand(self.nb_states, self.nb_actions)
        else:
            q = init_q*np.ones((self.nb_states, self.nb_actions))
        # Number of trajectories until now
        nb_trajs = 0
        # Number of collected samples
        nb_samples = 0
        # Global performance vector Jekyll/Hyde weighted average
        perf_vect = []
        # Jekyll performance vector (different from global only with Jekyll/Hyde algorithm)
        perf_jekyll_vect = []
        # UCB value of Hyde's policy (used only with Jekyll/Hyde algorithm)
        q_UCB = np.ones((self.nb_states, self.nb_actions))
        # transition counts
        counts = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        counts_jek = np.zeros((self.nb_states, self.nb_actions))
        counts_hyd = np.zeros((self.nb_states, self.nb_actions))
        # state initialization
        s = 0
        # length of current trajectory
        traj_length = 0
        # sum of gradient weights (used to make sure that each algorithm received the same amount of gradient)
        sum_factor = 0
        # controlling policy (used only with Jekyll/Hyde algorithm)
        pi_control = "Jekyll"
        # mle estimates of the transition function
        mle_transitions = np.zeros((self.nb_states, self.nb_actions, self.nb_states))

        st0 = 0
        st1 = 0
        st2 = 0
        st3 = 0
        st4 = 0
        st5 = 0

        count_jek = 1
        count_hyde = 0
        nb_samples_jek = 0
        nb_samples_hyd = 0

        while nb_trajs < self.max_nb_it:
            # update of the probability of selecting "Jekyll" vs "Hyde" (used only with Jekyll/Hyde algorithm)
            p_hyde = min(1, hyde_param / (nb_samples+1)**alpha)
            p_offpol = min(1, offpol_param / (nb_samples+1)**offpol_alpha)

            # Chooses next action
            if pi_control =="Jekyll":
                a = self.perform_action(s, pi)
            elif pi_control =="Hyde":
                a_s = np.argwhere(q_UCB[s]==q_UCB[s].max()).flatten()
                a = a_s[np.random.randint(len(a_s))]
            
            # Performs action
            r, sp, t = self.env.step(s, a)

            # Update counts and transition function
            counts[s,a,sp] += 1
            if pi_control =="Jekyll":
                counts_jek[s,a] += 1
            elif pi_control =="Hyde":
                counts_hyd[s,a] += 1
            mle_transitions[s,a] = counts[s,a]/counts[s,a].sum()
            minibatch = []

            if critic_type == 'sample-replay':
                for _ in range(minibatch_size):
                    tau = {}
                    if discounting == 'jh' and np.random.rand() < p_offpol and counts_hyd.sum()>0:
                        tau['s'], tau['a'] = sample_from(counts_hyd)
                        nb_samples_hyd += 1
                    else:
                        tau['s'], tau['a'] = sample_from(counts_jek)
                        nb_samples_jek += 1
                    tau['sp'] = choice(np.arange(self.nb_states), mle_transitions[tau['s'], tau['a']])
                    tau['r'] = self.env.reward_function[tau['s'], tau['sp']]
                    minibatch.append(tau)

            # CRITIC UPDATE:
            if critic_type == 'sarsa':
                # SARSA critic
                q[s,a] += critic_stepsize * (r + (1-t)*self.gamma*np.inner(q[sp],pi[sp]) - q[s,a])
            elif critic_type == 'qlearning':
                # Q-learning critic
                q[s,a] += critic_stepsize * (r + (1-t)*self.gamma*np.max(q[sp]) - q[s,a])
            elif critic_type == 'ucb': # (SARSA type)
                # r_ucb = min(1, np.sqrt(np.log(nb_trajs+1)/(0.001 + counts[s,a].sum())))
                r_ucb = np.sqrt(np.log(nb_trajs+1)/(0.001 + counts[s,a].sum()))
                # Q-learning critic
                q[s,a] += critic_stepsize * (r + critic_UCB_stepsize*r_ucb + (1-t)*self.gamma*np.inner(q[sp],pi[sp]) - q[s,a])
            elif critic_type == 'sample-replay':
                # SARSA with sample replay
                for tau in minibatch:
                    term = (tau['sp'] == self.env.final_state)
                    tderror = tau['r'] + (1-term)*self.gamma*np.inner(q[tau['sp']],pi[tau['sp']]) - q[tau['s'], tau['a']]
                    q[tau['s'], tau['a']] += critic_stepsize * tderror / minibatch_size
            elif critic_type == 'mle':
                # Computes the exact value of the main policy in the MLE MDP
                M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', mle_transitions, pi).reshape(nb_sa, nb_sa)
                d_pi = np.linalg.inv(M)
                q = np.dot(d_pi, self.R).reshape(self.nb_states,self.nb_actions)
                
            # UCB CRITIC UPDATE: similar to the other critic update (used only with Jekyll/Hyde algorithm)
            if discounting == 'jh' or discounting == 'jh2':
                r_ucb = np.sqrt(np.log(nb_trajs+1)/(0.001 + counts[s,a].sum()))
                if critic_type == 'sarsa':
                    # SARSA critic
                    q_UCB[s,a] += critic_UCB_stepsize * (r_ucb + (1-t)*self.gamma*np.inner(q_UCB[sp],pi[sp]) - q_UCB[s,a])
                    q_UCB[s,a] = q_UCB[s,a]
                elif critic_type == 'qlearning':
                    # Q-learning critic
                    q_UCB[s,a] += critic_UCB_stepsize * (r_ucb + (1-t)*self.gamma*np.max(q_UCB[sp]) - q_UCB[s,a])
                    q_UCB[s,a] = q_UCB[s,a]
                elif critic_type == 'sample-replay':
                    # SARSA with sample replay
                    for tau in minibatch:
                        r_tau = np.sqrt(np.log(nb_trajs+1)/(0.001 + counts[tau['s'], tau['a']].sum()))
                        term = (tau['sp'] == self.env.final_state)
                        tderror = r_tau + (1-term)*self.gamma*np.max(q_UCB[tau['sp']]) - q_UCB[tau['s'], tau['a']]
                        q_UCB[tau['s'], tau['a']] += critic_UCB_stepsize * tderror / minibatch_size
                        q_UCB[tau['s'], tau['a']] = min(1, q_UCB[tau['s'], tau['a']])
                elif critic_type == 'mle':
                    # Computes the UCB policy
                    pi_UCB = np.zeros((self.nb_states, self.nb_actions)) 
                    a_s = q_UCB.argmax(axis=1)
                    pi_UCB[np.arange(self.nb_states),a_s] = 1
                    # Computes the current UCB reward function
                    r_UCB = np.sqrt(1/(0.001 + counts.sum(axis=2)))
                    r_UCB[self.env.final_state] = 0
                    # Computes the exact value of the UCB policy in the UCB MLE MDP  
                    M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', mle_transitions, pi_UCB).reshape(nb_sa, nb_sa)
                    d_pi_UCB = np.linalg.inv(M)
                    q_UCB = np.dot(d_pi_UCB, r_UCB.reshape(nb_sa)).reshape(self.nb_states,self.nb_actions)
            
            # ACTOR UPDATE
            if discounting == 'theory':
                factor = self.gamma**traj_length
            if discounting == 'jh':
                factor = self.gamma**traj_length
            if discounting == 'jh2':
                factor = 1
            elif discounting == 'practice':
                factor = 1
            # DEPRECATED (won't be in the paper)
            # elif discounting == 'supercounted':
            #     factor = self.gamma**(-traj_length)
            # DEPRECATED (won't be in the paper)
            # elif discounting == 'cumul': # gamma'=gamma
            #     factor = self.gamma*self.gamma**traj_length
            # DEPRECATED (won't be in the paper)
            # elif discounting == 'cumul2': # gamma'=1
            #     factor = 1-self.gamma**(traj_length+1)
            # DEPRECATED (won't be in the paper)
            # elif discounting == 'linear': 
            #     factor = traj_length+1

            # Rescale updates on the same weight
            sum_factor += factor
            factor *= nb_trajs/sum_factor
            

            if parametrization == 'softmax':
                if discounting == 'jh2':
                    entropy = 0
                    if lambada>0:
                        entropy = -lambada*np.log(pi)# Update the parameters
                    theta += actor_stepsize * factor * pi * (q+entropy-np.sum(q*pi, axis=1).reshape(self.nb_states,1)) / self.nb_states
                    # For numerical stability, rescale the thetas.
                    theta -= theta.max(axis=1).reshape(self.nb_states,1)
                    # Update the main policy
                    pi = softmax(theta)
                elif critic_type == 'sample-replay':
                    for tau in minibatch:
                        # Entropy calculation (if used as a regularization)
                        entropy = 0
                        if lambada>0:
                            entropy = -lambada*np.log(pi[tau['s']])# Update the parameters
                        theta[tau['s']] += actor_stepsize * factor * pi[tau['s']] * (q[tau['s']]+entropy-np.inner(q[tau['s']],pi[tau['s']]))/minibatch_size
                        # For numerical stability, rescale the thetas.
                        theta[tau['s']] -= theta[tau['s']].max()
                        # Update the main policy
                        pi = softmax(theta)
                elif discounting == 'jh' and critic_type == 'mle' and critic_stepsize > 100:
                    entropy = 0
                    if lambada>0:
                        entropy = -lambada*np.log(pi)# Update the parameters
                    density = (1-p_offpol) * d_pi[0].reshape(self.nb_states,self.nb_actions).sum(axis=1).reshape(self.nb_states,1)/d_pi[0].sum() + p_offpol/self.nb_states
                    theta += actor_stepsize * factor * pi * (q+entropy-np.sum(q*pi, axis=1).reshape(self.nb_states,1)) * density
                    # For numerical stability, rescale the thetas.
                    theta -= theta.max(axis=1).reshape(self.nb_states,1)
                    # Update the main policy
                    pi = softmax(theta)
                else:
                    # Entropy calculation (if used as a regularization)
                    entropy = 0
                    if lambada>0:
                        entropy = -lambada*np.log(pi[s])# Update the parameters
                    theta[s] += actor_stepsize * factor * pi[s] * (q[s]+entropy-np.inner(q[s],pi[s]))
                    # For numerical stability, rescale the thetas.
                    theta[s] -= theta[s].max()
                    # Update the main policy
                    pi = softmax(theta)
            elif parametrization == 'direct':
                # Gradient (I update for all actions in a, but I maybe should update with q[s,a]/pi[s,a]?)
                # we could normalise with len(np.where(counts.sum(axis=2).sum(axis=1)>0)[0]) too
                if discounting == 'jh2':
                    pi += actor_stepsize * factor * q / self.nb_states
                    pi = proj_simplex.projection_simplex_sort(pi) 
                elif critic_type == 'sample-replay':
                    for tau in minibatch:
                        pi[tau['s']] += actor_stepsize * factor * q[tau['s']]
                        pi[tau['s']] = proj_simplex.projection_simplex_sort(pi[tau['s']].reshape(1, self.nb_actions)) 
                else:
                    pi[s] += actor_stepsize * factor * q[s]
                    pi[s] = proj_simplex.projection_simplex_sort(pi[s].reshape(1, self.nb_actions))
                    
            # Update current state and counters
            s = sp
            traj_length += 1
            nb_samples += 1

            # TRAJECTORY TERMINATION
            if t or traj_length > 99:
                # Update counters
                nb_trajs += 1
                traj_length = 0
                # initialize the current state
                s = 0

                # Update the controlling policy (used only with Jekyll/Hyde algorithm)
                if discounting == 'jh' or discounting == 'jh2':
                    if np.random.rand() > p_hyde:
                        pi_control = "Jekyll"
                        count_jek +=1
                    else:
                        pi_control = "Hyde"
                        count_hyde +=1
                else:
                    pi_control = "Jekyll"

                if (nb_trajs - 1) % (self.max_nb_it//1000) == 0 or nb_trajs == self.max_nb_it:
                    # True value tracking of the main policy (aka Jekyll)
                    M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', self.P, pi).reshape(nb_sa, nb_sa)
                    sa_0 = np.zeros(nb_sa)
                    sa_0[:self.nb_actions] = pi[0]
                    d_pi = np.linalg.inv(M)
                    true_q = np.dot(d_pi, self.R)
                    perf = np.dot(true_q, sa_0)
                    perf_jekyll_vect.append(perf)

                    # True value tracking of Hyde's policy (used only with Jekyll/Hyde algorithm) 
                    if discounting == 'jh' or discounting == 'jh2':
                        pi_UCB = np.zeros((self.nb_states, self.nb_actions))
                        a_s = q_UCB.argmax(axis=1)
                        pi_UCB[np.arange(self.nb_states),a_s] = 1
                        sa_0[:self.nb_actions] = pi_UCB[0]
                        M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', self.P, pi_UCB).reshape(nb_sa, nb_sa)
                        d_pi = np.linalg.inv(M)
                        true_q_UCB = np.dot(d_pi, self.R)
                        perf_UCB = np.dot(true_q_UCB,sa_0)
                        perf = (1-p_hyde)*perf + p_hyde*perf_UCB
                    perf_vect.append(perf)

                    # Stops if a stopping criterion on performance has been met
                    if perf_jekyll_vect[-1] > self.perf_stop and perf > self.perf_stop:
                        print(nb_trajs)
                        return nb_trajs
            
        
        # Returns the maximal value if a stopping criterion on performance has been met 
        if self.perf_stop < 1:
            return self.max_nb_it
        if discounting == 'jh' or discounting == 'jh2':
            return perf_vect, perf_jekyll_vect
        return perf_vect
