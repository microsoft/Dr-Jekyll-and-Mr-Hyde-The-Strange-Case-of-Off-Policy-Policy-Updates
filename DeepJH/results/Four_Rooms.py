import argparse
import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.JH_discrete import JH_Discrete
from agents.DQN_agents.DDQN import DDQN
from environments.Four_Rooms_Environment import Four_Rooms_Environment
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--runs', default=1, type=int)
parser.add_argument('--num_episodes', default=500, type=int)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--level', default=1, type=int)
parser.add_argument('--agent', default='JH_Discrete', type=str)
parser.add_argument('--sto', default=0, type=float)
parser.add_argument('--dqn_lr', default=0.01, type=float)
parser.add_argument('--sac_lr', default=0.01, type=float)
parser.add_argument('--jekyll_lr', default=0.001, type=float)
parser.add_argument('--rnd_lr', default=0.003, type=float)
parser.add_argument('--rnd_weight', default=1.0, type=float)
parser.add_argument('--normalize_rnd', action='store_true')
parser.add_argument('--no_rnd_actions', action='store_true')
parser.add_argument('--discount', default=0.9, type=float)
args = parser.parse_args()


config = Config()
if args.seed is not None:
    config.seed = args.seed
    config.randomise_random_seed = False
else:
    config.seed = 1 # args.seed
    config.randomise_random_seed = True
config.runs_per_agent = args.runs
config.num_episodes_to_run = args.num_episodes
config.log_path = 'results'
config.file_to_save_data_results = os.path.join(config.log_path, "Four_Rooms_Static.pkl")
config.file_to_save_results_graph = os.path.join(config.log_path, "Four_Rooms_Static.png")
config.use_GPU = True
config.GPU_id = args.gpu
config.logging_interval = 10

height = 15
width = 15
random_goal_place = False
num_possible_states = (height * width) ** (1 + 1*random_goal_place)
embedding_dimensions = [[num_possible_states, 20]]

print("Num possible states ", num_possible_states)

config.environment = Four_Rooms_Environment(
    height, width, stochastic_actions_probability=args.sto, random_start_user_place=True,
    random_goal_place=random_goal_place, level=args.level)

config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.overwrite_existing_results_file = False
config.save_model = False
do_evaluation_iterations = True

if 'RND' in args.agent:
    agent = args.agent[:-3]
    rnd_counts = True
else:
    agent = args.agent
    rnd_counts = False

if 'JH' in args.agent:
    batch_size = 128
else:
    batch_size = 256

if args.level == 1:
    epsilon_scale = 4000
    epsilon_decay_rate_denominator = 10
else:
    epsilon_scale = 40000
    epsilon_decay_rate_denominator = 100

rnd_columns_of_data_to_be_embedded = [0]
rnd_embedding_dimensions = embedding_dimensions

config.hyperparameters = {

    "DQN_Agents": {
        "linear_hidden_units": [30, 10],
        "learning_rate": args.dqn_lr,
        "buffer_size": 40000,
        "batch_size": batch_size,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": embedding_dimensions,
        "gradient_clipping_norm": 5,
        "update_every_n_steps": 1,
        "epsilon_decay_rate_denominator": epsilon_decay_rate_denominator,  # TODO
        "discount_rate": args.discount,
        "tau": 0.01,
        "clip_rewards": False,
        "do_evaluation_iterations": do_evaluation_iterations,
        "exploration_bonus": rnd_counts,
        "exploration_bonus_weight": args.rnd_weight,

        "Explorer": {
            "learning_rate": args.rnd_lr,
            "discount_rate": args.discount,
            "features_size": 20,
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": embedding_dimensions,
            "linear_hidden_units": [20, 20],
            "target_linear_hidden_units": [20],
            "final_layer_activation": "None",
            "gradient_clipping_norm": 0.7,
            "scale": 1,
            "rnd_actions": not args.no_rnd_actions,
        }
    },

    "Actor_Critic_Agents": {

        "min_steps_before_learning": 400,
        "batch_size": batch_size,
        "discount_rate": args.discount,
        "update_every_n_steps": 1,
        "do_evaluation_iterations": do_evaluation_iterations,
        "clip_rewards": False,
        # SAC entropy
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": 0.0,
        "add_extra_noise": False,
        # Decay of epsilon for JH
        "epsilon_scale": epsilon_scale,
        "exploration_bonus": rnd_counts,
        "exploration_bonus_weight": args.rnd_weight,

        "Actor": {
            "learning_rate": args.sac_lr if args.agent == 'SAC_Discrete' else args.jekyll_lr,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": embedding_dimensions,
            "tau": 0.01,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": args.sac_lr if args.agent == 'SAC_Discrete' else args.jekyll_lr,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": embedding_dimensions,
            "buffer_size": 40000,
            "tau": 0.01,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Hyde": {
            "linear_hidden_units": [30, 10],
            "learning_rate": args.dqn_lr,
            "buffer_size": 40000,
            "batch_size": 128,
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": embedding_dimensions,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 10,
            "discount_rate": args.discount,
            "learning_iterations": 1,
            "tau": 0.01,
            "clip_rewards": False
        },

        "Explorer": {
            "learning_rate": args.rnd_lr,
            "discount_rate": args.discount,
            "batch_size": 128,
            "features_size": 20,
            "columns_of_data_to_be_embedded": rnd_columns_of_data_to_be_embedded,
            "embedding_dimensions": rnd_embedding_dimensions,
            "linear_hidden_units": [20, 20],
            "target_linear_hidden_units": [20],
            "final_layer_activation": "None",
            "gradient_clipping_norm": 0.7,
            "scale": 1,
            "normalize_rnd": args.normalize_rnd,
            "rnd_actions": not args.no_rnd_actions,
        }
    }
}

if __name__== '__main__':

    if args.agent:
        AGENTS = [eval(agent)]
    else:
        AGENTS = [JH_Discrete, DDQN, SAC_Discrete]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
