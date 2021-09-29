from collections import Counter

import copy
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent
from agents.actor_critic_agents.explorer import Explorer
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer


class DQN(Base_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed, self.device)
        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"], eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)
        self.nature = config.nature if hasattr(config, 'nature') else False
        print('Is nature: ', self.nature)

        self.exploration_bonus = self.hyperparameters.get("exploration_bonus", False)
        if 'Explorer' in self.hyperparameters:
            '''
            Explorer. Either Random Network Distillation or count based. Creates intrinsic rewards for Hyde to explore properly
            '''
            explorer_config = copy.deepcopy(self.config)
            explorer_config.hyperparameters = explorer_config.hyperparameters['Explorer']
            self.rnd_actions = self.hyperparameters['Explorer'].get('rnd_actions', False)
            if isinstance(self.state_size, int):
                self.explorer = Explorer(explorer_config)
            else:
                state_size = torch.Size((1,) + self.state_size[1:])
                self.explorer = Explorer(
                    explorer_config, state_size=state_size)
        self.exploration_bonus_weight = self.hyperparameters.get("exploration_bonus_weight", 0)
        if self.exploration_bonus:
            print('USING EXPLORATION BONUS')
        self.training_started = False

    def reset_game(self):
        super(DQN, self).reset_game()
        if not self.nature:
            self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        eval_ep = self.episode_number % self.training_episodes_per_eval_episode == 0 and self.do_evaluation_iterations
        if eval_ep:
            self.evaluate()
            return
        while not self.done:
            self.action = self.pick_action(eval_ep)
            self.explorer.log_state_action(self.state, self.action)
            self.conduct_action(self.action)
            if self.time_for_q_network_to_learn():
                self.training_started = True
                for _ in range(self.hyperparameters.get("learning_iterations", 1)):
                    self.learn()
            if not eval_ep:
                self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += 1
            if self.nature and self.global_step_number % self.hyperparameters.get("target_update_frequency", 40000) == 0:
                self.hard_update_of_target_network(self.q_network_local, self.q_network_target)
        if eval_ep: self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None:
            memory = self.memory
        if experience is None:
            experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience, torch.ones([1, 1]))

    def pick_action(self, eval_ep=False, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1: state = state.unsqueeze(0)

        self.q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train() #puts network back in training mode
        if self.rnd_actions and self.training_started:
            # Trick. Compute intrinsic rewards for all actions. Pick best between that and q hyde
            rewards = self.explorer.compute_rewards_all_actions(state)
            action_values = torch.max(torch.cat((action_values.unsqueeze(2), rewards.unsqueeze(2)), dim=2), dim=2)[0]

        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration or eval_ep,
                                                                                    "episode_number": self.episode_number,
                                                                                    "steps": self.global_step_number})
        self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        return action

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences() # Sample experiences
        else: states, actions, rewards, next_states, dones = experiences

        rnd_rewards = None
        if self.exploration_bonus and self.rnd_actions:
            # Compute intrinsic rewards
            rnd_rewards = self.explorer.compute_intrinsic_reward_and_learn(states, actions=actions)
        elif self.exploration_bonus:
            # Train the explorer on the transitions with intrinsic RND rewards (and train RND)
            rnd_rewards = self.explorer.compute_intrinsic_reward_and_learn(next_states)

        loss = self.compute_loss(states, next_states, rewards, actions, dones, rnd_rewards=rnd_rewards)

        actions_list = [action_X.item() for action_X in actions ]

        self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones, rnd_rewards=None):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(
                next_states, rewards, dones, rnd_rewards=rnd_rewards)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones, rnd_rewards=None):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones, rnd_rewards=rnd_rewards)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones, rnd_rewards=None):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        if rnd_rewards is not None:
            Q_targets_current += self.exploration_bonus_weight * rnd_rewards
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.q_network_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones
