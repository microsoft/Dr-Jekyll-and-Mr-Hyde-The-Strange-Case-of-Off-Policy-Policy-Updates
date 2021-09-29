from agents.Base_Agent import Base_Agent
from agents.DQN_agents.DDQN import DDQN
from agents.actor_critic_agents.explorer import Explorer
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from utilities.Utility_Functions import create_actor_distribution

import copy
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.optim import Adam


TRAINING_EPISODES_PER_EVAL_EPISODE = 10


class JH_Discrete(Base_Agent):
    """Jekyll and Hyde"""
    agent_name = "JH"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "DISCRETE", "Action types must be discrete."
        assert self.config.hyperparameters["Actor"]["final_layer_activation"] == "Softmax", "Final actor layer must be softmax"
        self.hyperparameters = config.hyperparameters

        '''
        Jekyll Agent. 2 q-value critics + one actor + its own memory
        '''
        self.jekyll_critic_local = self.create_NN(
            input_dim=self.state_size, output_dim=self.action_size, key_to_use="Critic")
        self.jekyll_critic_local_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                                    key_to_use="Critic", override_seed=self.config.seed + 1)
        self.jekyll_critic_optimizer = Adam(self.jekyll_critic_local.parameters(),
                                            lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.jekyll_critic_optimizer_2 = Adam(self.jekyll_critic_local_2.parameters(),
                                              lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.jekyll_critic_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                                   key_to_use="Critic")
        self.jekyll_critic_target_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                                     key_to_use="Critic")
        Base_Agent.copy_model_over(
            self.jekyll_critic_local, self.jekyll_critic_target)
        Base_Agent.copy_model_over(
            self.jekyll_critic_local_2, self.jekyll_critic_target_2)
        self.jekyll_memory = Replay_Buffer(
            self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"], self.config.seed, device=self.device)
        self.jekyll_actor = self.create_NN(
            input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.jekyll_actor_optimizer = Adam(self.jekyll_actor.parameters(),
                                           lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)

        '''
        Explorer. Either Random Network Distillation or count based. Creates intrinsic rewards for Hyde to explore properly
        '''
        explorer_config = copy.deepcopy(self.config)
        explorer_config.hyperparameters = explorer_config.hyperparameters['Explorer']
        if isinstance(self.state_size, int):
            self.explorer = Explorer(explorer_config)
        else:
            state_size = torch.Size((1,) + self.state_size[1:])
            self.explorer = Explorer(
                explorer_config, state_size=state_size)
        self.rnd_actions = self.hyperparameters['Explorer'].get(
            'rnd_actions', False)

        '''
        Hyde agent. DQN trained on rewards from RND. Used to explore
        '''
        dqn_config = copy.deepcopy(self.config)
        dqn_config.hyperparameters = dqn_config.hyperparameters['Hyde']
        self.hyde = Hyde(dqn_config)

        '''
        Exploration parameters
        '''
        # Flag to determine whether to use hyde when stepping in the environment (and store the transition in the corresponding buffer)
        # Full exploration at first
        self._is_hyde = True
        self.epsilon_scale = self.hyperparameters['epsilon_scale']

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)

    @property
    def has_hyde(self):
        return True

    @property
    def is_hyde(self):
        return self._is_hyde

    @property
    def epsilon(self):
        if self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            return 1
        return 1 / (1 + (self.global_step_number - self.hyperparameters["min_steps_before_learning"]) / self.epsilon_scale)

    @property
    def d(self):
        return 0.5

    def set_is_hyde(self):
        self._is_hyde = random.random() < self.epsilon

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        if eval_ep:
            self.evaluate()
            return
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.explorer.log_state_action(self.state, self.action)
            self.conduct_action(self.action)
            if self.time_for_agents_to_learn():
                for _ in range(self.hyperparameters.get("learning_updates_per_learning_session", 1)):
                    self.learn()
            if hasattr(self.environment, '_max_episode_steps'):
                mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            else:
                mask = False if self.episode_step_number_val >= 27000 else self.done
            if not eval_ep:
                self.save_experience(experience=(
                    self.state, self.action, self.reward, self.next_state, mask))
            self.state = self.next_state
            self.global_step_number += 1
        # Decide who to follow
        self.set_is_hyde()
        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

    def save_experience(self, experience):
        """Saves the recent experience to the memory buffer"""
        if self.is_hyde or len(self.hyde.memory) <= self.hyperparameters["batch_size"]:
            self.hyde.memory.add_experience(*experience, torch.ones([1, 1]))
        self.jekyll_memory.add_experience(*experience, torch.ones([1, 1]))

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None:
            state = self.state
        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = self.environment.action_space.sample()
        else:
            action = self.actor_pick_action(state=state)
        return action

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None:
            state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            if self.is_hyde and not eval:
                action = self.hyde_pick_action(state)
            else:
                action, _ = self.jekyll_pick_action(state, greedy=eval)
        action = action.detach().cpu().numpy()
        return action[0]

    def jekyll_pick_action(self, state, greedy=False):
        """Given the state, return an action, either sampled or the argmax and the action probabilities"""
        action_probabilities = self.jekyll_actor(state)
        if greedy:
            return torch.argmax(action_probabilities, dim=-1), action_probabilities
        action_distribution = create_actor_distribution(
            self.action_types, action_probabilities, self.action_size)
        return action_distribution.sample().cpu(), action_probabilities

    def hyde_pick_action(self, state):
        self.hyde.q_network_local.eval()  # puts network in evaluation mode
        action_values = self.hyde.q_network_local(state)
        self.hyde.q_network_local.train()  # puts network back in training mode
        return torch.argmax(action_values, dim=-1)

    def time_for_agents_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
            self.enough_experiences_to_learn_from(
        ) and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        if self.is_hyde:
            return len(self.hyde.memory) > self.hyperparameters["batch_size"]
        return len(self.jekyll_memory) > self.hyperparameters["batch_size"]

    def learn(self):
        """Runs a learning iteration for Jekyll, Hyde and the random network distillation"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, _ = self.sample_experiences()

        if self.has_hyde:
            # Train Hyde on the transitions with intrinsic RND rewards (and train RND)
            if self.rnd_actions:
                # Compute intrinsic rewards
                rnd_reward_batch = self.explorer.compute_intrinsic_reward_and_learn(state_batch, actions=action_batch)
                # Learn from everything
                self.hyde.learn(state_batch, action_batch, rnd_reward_batch, next_state_batch)
            else:
                rnd_reward_batch = self.explorer.compute_intrinsic_reward_and_learn(next_state_batch)
                self.hyde.learn(state_batch, action_batch, rnd_reward_batch, next_state_batch)

        # # Train Jekyll
        qf1_loss, qf2_loss, min_qf = self.calculate_jekyll_critic_losses(
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        self.update_jekyll_critic_parameters(qf1_loss, qf2_loss)

        policy_loss = self.calculate_jekyll_actor_loss(state_batch, min_qf)
        self.update_jekyll_actor_parameters(policy_loss)

    def sample_experiences(self):
        h_states, h_actions, h_rewards, h_next_states, h_dones, h_int_learn_batch = self.hyde.memory.sample_with_extra()
        j_states, j_actions, j_rewards, j_next_states, j_dones, j_int_learn_batch = self.jekyll_memory.sample_with_extra()
        return torch.cat((h_states, j_states), dim=0), \
            torch.cat((h_actions, j_actions), dim=0), \
            torch.cat((h_rewards, j_rewards), dim=0), \
            torch.cat((h_next_states, j_next_states), dim=0), \
            torch.cat((h_dones, j_dones), dim=0), \
            torch.cat((h_int_learn_batch, j_int_learn_batch), dim=0)

    def calculate_jekyll_actor_loss(self, state_batch, qf):
        """Calculates the loss for the jekyll actor"""
        _, action_probabilities = self.jekyll_pick_action(state_batch)
        advantages = (qf - action_probabilities * qf).detach()
        jekyll_actor_loss = -1.0 * action_probabilities * advantages
        jekyll_actor_loss = jekyll_actor_loss.mean()
        return jekyll_actor_loss

    def calculate_jekyll_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss.
        Also returns the q values, required to compute the advantage in the jekyll actor loss"""
        with torch.no_grad():
            _, action_probabilities = self.jekyll_pick_action(
                next_state_batch)
            qf1_next_target = self.jekyll_critic_target(next_state_batch)
            qf2_next_target = self.jekyll_critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * \
                torch.min(qf1_next_target, qf2_next_target)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + \
                (1.0 - mask_batch) * \
                self.hyperparameters["discount_rate"] * (min_qf_next_target)

        qf1 = self.jekyll_critic_local(state_batch)
        qf2 = self.jekyll_critic_local_2(state_batch)
        min_qf_local = torch.min(qf1, qf2).detach()
        qf1_loss = F.mse_loss(qf1.gather(1, action_batch.long()), next_q_value)
        qf2_loss = F.mse_loss(qf2.gather(1, action_batch.long()), next_q_value)
        return qf1_loss, qf2_loss, min_qf_local

    def update_jekyll_critic_parameters(self, critic_loss_1, critic_loss_2):
        """Updates the parameters for both jekyll critics"""
        self.take_optimisation_step(self.jekyll_critic_optimizer, self.jekyll_critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.jekyll_critic_optimizer_2, self.jekyll_critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.jekyll_critic_local, self.jekyll_critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.jekyll_critic_local_2, self.jekyll_critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_jekyll_actor_parameters(self, actor_loss):
        """Updates the parameters for the jekyll actor"""
        self.take_optimisation_step(self.jekyll_actor_optimizer, self.jekyll_actor, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("----------------------------")


class Hyde(DDQN):
    """A double DQN agent"""
    agent_name = "DDQN"

    def __init__(self, config):
        DDQN.__init__(self, config)

    def learn(self, states, actions, rewards, next_states):
        """Runs a learning iteration for the Q network"""

        loss = self.compute_loss(states, next_states, rewards, actions)

        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local,
                                    loss, self.hyperparameters["gradient_clipping_norm"])

        self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
                                           self.hyperparameters["tau"])  # Update the target network

    def compute_loss(self, states, next_states, rewards, actions,):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        return rewards + self.hyperparameters["discount_rate"] * Q_targets_next
