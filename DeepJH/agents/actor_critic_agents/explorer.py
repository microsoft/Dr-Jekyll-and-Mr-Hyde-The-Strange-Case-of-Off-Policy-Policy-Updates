from multiprocessing.dummy import Value
from agents.Base_Agent import Base_Agent

import copy
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, device, epsilon=1e-4, shape=()):
        self.device = device
        self.mean = torch.zeros(shape).to(self.device)
        self.var = torch.ones(shape).to(self.device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, axis=0)
        batch_var = torch.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count * \
            batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Explorer(Base_Agent):
    """Random Network Distillation or count based. Not really an agent"""
    agent_name = "Explorer"

    def __init__(self, config, state_size=None):
        Base_Agent.__init__(self, config, log_info=False, state_size=state_size)
        self.hyperparameters = config.hyperparameters
        self.count_based = self.hyperparameters.get('count_based', False)
        self.scale = self.hyperparameters.get('scale', 1)
        self.normalize = self.hyperparameters.get('normalize_rnd', False)
        self.batch_size = self.hyperparameters.get('batch_size', 1)
        self.rnd_actions = self.hyperparameters.get('rnd_actions', False)
        self.state_actions = torch.zeros((225, self.action_size)).to(self.device)
        if self.count_based:
            self.visited_states = torch.ones((225,)).to(self.device)
            self.visited_state_actions = torch.ones((225, self.action_size)).to(self.device)
        else:
            if self.rnd_actions:
                self.predictors, self.targets, self.optimizers = [], [], []
                for i in range(self.action_size):
                    predictor, target, optimizer = self._network_factory()
                    if i > 0:
                        self.copy_model_over(self.predictors[0], predictor)
                    self.predictors.append(predictor)
                    self.targets.append(target)
                    self.optimizers.append(optimizer)
            else:
                self.predictor, self.target, self.optimizer = self._network_factory()

        self.steps = 0
        self.reward_rms = RunningMeanStd(self.device)
        self.losses = []
        self.states_so_far_np = []
        self._init_states_so_far()

    def _init_states_so_far(self, action=None):
        if self.rnd_actions:
            if action:
                self.states_so_far[action] = torch.zeros([0, 1]).to(self.device)
            else:
                self.states_so_far = [torch.zeros([0, 1]).to(self.device) for _ in range(self.action_size)]
        else:
            self.states_so_far = torch.zeros([0, 1]).to(self.device)

    def _network_factory(self):
        predictor = self.create_NN(
            input_dim=self.state_size, output_dim=self.hyperparameters['features_size'],
            override_seed=self.config.seed + 1, hyperparameters=self.hyperparameters)
        target_hyperparameters = copy.deepcopy(self.hyperparameters)
        target_hyperparameters["linear_hidden_units"] = target_hyperparameters["target_linear_hidden_units"]
        target = self.create_NN(
            input_dim=self.state_size, output_dim=self.hyperparameters['features_size'])

        optimizer = Adam(predictor.parameters(),
                         lr=self.hyperparameters["learning_rate"], eps=1e-4)

        return predictor, target, optimizer

    def log_state_action(self, state, action):
        self.state_actions[state[0], action] += 1

    def compute_intrinsic_reward_and_learn(self, states, learn=True, actions=None, int_learn_batch=None):
        self.steps += 1
        if states.ndim == 4:
          # Only get last observation for RND
          states = torch.unsqueeze(states[:, -1, :, :], 1)
        # Get rewards
        rewards = self.compute_intrinsic_reward(states, learn=learn, actions=actions, int_learn_batch=int_learn_batch)
        if self.normalize:
            mean, std, count = torch.mean(rewards), torch.std(rewards), len(rewards)
            self.reward_rms.update_from_moments(mean, std ** 2, count)
            rewards /= torch.sqrt(self.reward_rms.var)

        return rewards.reshape((-1, 1))

    def compute_intrinsic_reward(self, states, learn=True, actions=None, int_learn_batch=None):
        if self.count_based:
            return self.compute_counts(states, learn=learn, actions=actions, int_learn_batch=int_learn_batch)
        else:
            return self.compute_preds(states, learn=learn, actions=actions, int_learn_batch=int_learn_batch)

    def compute_counts(self, states, learn=True, actions=None, int_learn_batch=None):
        if actions is not None:
            if not self.rnd_actions:
                raise ValueError
            rewards = 1 / torch.sqrt(self.visited_state_actions[states.long().squeeze(1), actions.long().squeeze(1)]).unsqueeze(1)
            if learn:
                indices = torch.where(int_learn_batch == 1)[0]
                states_to_learn = states[indices]
                actions_to_learn = actions[indices]
                self.learn_counts(states_to_learn, actions_to_learn)
        else:
            rewards = 1 / torch.sqrt(self.visited_states[states.long()])
            if learn:
                states_to_learn = states[torch.where(int_learn_batch == 1)[0]]
                self.learn_counts(states_to_learn)
        return rewards

    def get_counts_all_actions(self, state):
        return self.state_actions[state[0].long()]

    def compute_preds(self, states, learn=True, actions=None, int_learn_batch=None):
        if actions is not None:
            if not self.rnd_actions:
                raise ValueError
            intrinsic_reward = torch.zeros((len(states), 1)).to(self.device)
            for i in range(self.action_size):
                indices = torch.where(actions == i)[0]
                states_per_action = states[indices]
                target_next_feature = self.targets[i](states_per_action)
                predict_next_feature = self.predictors[i](states_per_action)
                intrinsic_reward[indices] = self._compute_intrinsic_reward(target_next_feature, predict_next_feature).unsqueeze(1)
                if learn:
                    self.states_so_far[i] = torch.cat(
                        (self.states_so_far[i], states_per_action))
                    self.learn_pred(index_to_train=i)
        else:
            target_next_feature = self.target(states)
            predict_next_feature = self.predictor(states)
            intrinsic_reward = self._compute_intrinsic_reward(target_next_feature, predict_next_feature)
        if learn and actions is None:
            self.learn_pred(predict_next_feature=predict_next_feature, target_next_feature=target_next_feature)
        return self.scale * intrinsic_reward

    @staticmethod
    def _compute_intrinsic_reward(target_feature, predict_feature):
        return ((target_feature - predict_feature).pow(2).sum(1) / 2).detach()

    def learn(self, states):
        """ Minimize the mse loss between predictions and target"""
        if self.count_based:
            self.learn_counts(states)
        else:
            self.learn_pred(states)

    def learn_counts(self, states, actions=None):
        """ Update the visitation counts"""
        for i in range(len(states)):
            if actions is not None:
                self.visited_state_actions[int(states[i].item()), int(actions[i].item())] += 1
            else:
                self.visited_states[int(states[i].item())] += 1

    def learn_pred(self, states=None, predict_next_feature=None, target_next_feature=None, index_to_train=None):
        if predict_next_feature is None and states is None:
            if index_to_train is None:
                raise ValueError
            samples = self.states_so_far[index_to_train]
            target = self.targets[index_to_train]
            predictor = self.predictors[index_to_train]
            self._init_states_so_far(action=index_to_train)
            if len(samples) >= self.batch_size or index_to_train is not None:
                target_next_feature = target(samples)
                predict_next_feature = predictor(samples)
            else:
                return
        elif states is not None:
            target_next_feature = self.target(states)
            predict_next_feature = self.predictor(states)
        prediction_loss = F.mse_loss(predict_next_feature, target_next_feature.detach())
        self.losses.append(prediction_loss.item())
        self.update_rnd_parameters(prediction_loss, index_to_train=index_to_train)

    def update_rnd_parameters(self, prediction_loss, index_to_train=None):
        """Updates the parameters for the rnd"""
        if index_to_train is not None:
            self.take_optimisation_step(self.optimizers[index_to_train], self.predictors[index_to_train], prediction_loss,
                                        self.hyperparameters["gradient_clipping_norm"])
        else:
            self.take_optimisation_step(self.optimizer, self.predictor, prediction_loss,
                                        self.hyperparameters["gradient_clipping_norm"])
