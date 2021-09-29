import copy
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from agents.actor_critic_agents.SAC import SAC
from agents.actor_critic_agents.explorer import Explorer
from utilities.Utility_Functions import create_actor_distribution

class SAC_Discrete(SAC):
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""
    agent_name = "SAC"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "DISCRETE", "Action types must be discrete. Use SAC instead for continuous actions"
        assert self.config.hyperparameters["Actor"]["final_layer_activation"] == "Softmax", "Final actor layer must be softmax"
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                           key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                           key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                            key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)

        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]
        assert not self.hyperparameters["add_extra_noise"], "There is no add extra noise option for the discrete version of SAC at moment"
        self.add_extra_noise = False
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

        self.exploration_bonus = self.hyperparameters.get("exploration_bonus", False)
        self.exploration_bonus_weight = self.hyperparameters.get("exploration_bonus_weight", 0)
        explorer_config = copy.deepcopy(self.config)
        explorer_config.hyperparameters = explorer_config.hyperparameters['Explorer']
        self.rnd_actions = self.hyperparameters['Explorer'].get('rnd_actions', False)
        if isinstance(self.state_size, int):
            self.explorer = Explorer(explorer_config)
        else:
            state_size = torch.Size((1,) + self.state_size[1:])
            self.explorer = Explorer(
                explorer_config, state_size=state_size)
        if self.exploration_bonus:
            print('USING EXPLORATION BONUS')
            '''
            Explorer. Either Random Network Distillation or count based. Creates intrinsic rewards for Hyde to explore properly
            '''
            self.critic_rnd = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Critic")
            self.critic_rnd_optimizer = torch.optim.Adam(self.critic_rnd.parameters(),
                                                         lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()

        rnd_reward_batch, rewards_next = None, None
        if self.exploration_bonus and self.rnd_actions:
            # Compute intrinsic rewards
            rnd_reward_batch = self.explorer.compute_intrinsic_reward_and_learn(state_batch, actions=action_batch)
            # Compute intrinsic rewards at the next state
            rewards_next = self.explorer.compute_max_rewards_all_actions_states(next_state_batch, maxx=False)
        elif self.exploration_bonus:
            # Train the explorer on the transitions with intrinsic RND rewards (and train RND)
            rnd_reward_batch = self.explorer.compute_intrinsic_reward_and_learn(next_state_batch)

        qf1_loss, qf2_loss, rnd_loss = self.calculate_critic_losses(
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, rnd_reward_batch, rewards_next)
        self.update_critic_parameters(qf1_loss, qf2_loss, rnd_loss)

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None
        self.update_actor_parameters(policy_loss, alpha_loss)

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, rnd_reward_batch=None, rewards_next=None):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            _, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)

        qf1 = self.critic_local(state_batch)
        qf2 = self.critic_local_2(state_batch)
        qf1_loss = F.mse_loss(qf1.gather(1, action_batch.long()), next_q_value)
        qf2_loss = F.mse_loss(qf2.gather(1, action_batch.long()), next_q_value)

        if self.exploration_bonus and rnd_reward_batch is not None:
            with torch.no_grad():
                if rewards_next is not None:
                    qf_rnd_next = action_probabilities * torch.max(torch.cat((self.critic_rnd(next_state_batch).unsqueeze(2), rewards_next.unsqueeze(2)), dim=2), dim=2)[0]
                else:
                    qf_rnd_next = action_probabilities * self.critic_rnd(next_state_batch)
                qf_rnd_next = qf_rnd_next.sum(dim=1).unsqueeze(-1)
                next_q_rnd = rnd_reward_batch + self.hyperparameters["discount_rate"] * qf_rnd_next
            qf_rnd = self.critic_rnd(state_batch)
            rnd_loss = F.mse_loss(qf_rnd.gather(1, action_batch.long()), next_q_rnd)
        else:
            rnd_loss, qf_rnd = None, None

        return qf1_loss, qf2_loss, rnd_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        _, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        if self.exploration_bonus and self.exploration_bonus_weight:
            bonus = self.exploration_bonus_weight * self.critic_rnd(state_batch)
            inside_term = self.alpha * log_action_probabilities - min_qf_pi - bonus
        else:
            inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities

    def update_critic_parameters(self, critic_loss_1, critic_loss_2, rnd_loss):
        """Updates the parameters for both critics"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

        if self.exploration_bonus and rnd_loss is not None:
            self.take_optimisation_step(self.critic_rnd_optimizer, self.critic_rnd, rnd_loss,
                                        self.hyperparameters["Critic"]["gradient_clipping_norm"])
