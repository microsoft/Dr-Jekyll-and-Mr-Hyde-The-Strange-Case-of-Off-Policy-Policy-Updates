# authors: anonymous

import numpy as np
import garnets 
import chain
import gradient_ascent
import yaml
import pickle
from utils import prt

prt('Start of experiment')

def run_experiment(config_file):
    with open(config_file + '.cfg') as stream:
        cfg = yaml.safe_load(stream)
        prt('Config loaded from ' + config_file + '.cfg')

    if cfg['seed'] > 0:
        np.random.seed(cfg['seed'])

    for _ in range(cfg['nb_expes']):
        # for nb_states in range(3,26):
        #     expe = {}
        #     expe['cfg'] = cfg
        #     cfg['nb_states'] = nb_states
        # for vr in np.arange(100)/100:
        #     expe = {}
        #     expe['cfg'] = cfg
        #     cfg['vr'] = vr
        # for actor_stepsize in 10**(np.arange(6)/5 + 3):
        #     expe = {}
        #     expe['cfg'] = cfg
        #     prt(actor_stepsize)
        #     for algo in expe['cfg']['algos']:
        #         algo['actor_stepsize'] = actor_stepsize
        # for ucb_critic_stepsize in 10**(np.arange(17)/4 - 4):
        #     expe = {}
        #     expe['cfg'] = cfg
        #     prt(ucb_critic_stepsize)
        #     for algo in expe['cfg']['algos']:
        #         algo['critic_UCB_stepsize'] = ucb_critic_stepsize
        # for q_init in np.arange(101)/100:
        #     expe = {}
        #     expe['cfg'] = cfg
        #     prt('q_init = ' + str(q_init))
        #     for algo in expe['cfg']['algos']:
        #         algo['init_q'] = q_init
        # for offpol in np.arange(101)/100:
        #     expe = {}
        #     expe['cfg'] = cfg
        #     prt('offpol = ' + str(offpol))
        #     for algo in expe['cfg']['algos']:
        #         algo['offpol_param'] = offpol
        # for critic_stepsize in 10**(np.arange(16)/5 - 3):
        #     expe = {}
        #     expe['cfg'] = cfg
        #     prt(critic_stepsize)
        #     for algo in expe['cfg']['algos']:
        #         algo['critic_stepsize'] = critic_stepsize
        for hyde_param in np.arange(51)/100:
            expe = {}
            expe['cfg'] = cfg
            prt('hyde_param = ' + str(hyde_param))
            for algo in expe['cfg']['algos']:
                algo['hyde_param'] = hyde_param
        # for lambada in 10**(np.arange(51)/10 - 4):
        #     expe = {}
        #     expe['cfg'] = cfg
        #     prt(lambada)
        #     for algo in expe['cfg']['algos']:
        #         algo['lambada'] = lambada
        #     expe = {}
        #     expe['cfg'] = cfg
            expe['seed'] = np.random.randint(99999999)
            np.random.seed(expe['seed'])
            prt(cfg['nb_states'])

            # chain experiment
            if cfg['environment'] == 'chain':
                env = chain.Chain(cfg['nb_states'], stochasticity=cfg['stochasticity'], gamma=cfg['gamma'], value_ratio=cfg['vr'])
                p_star = cfg['gamma']**(cfg['nb_states']-2)
                p_rand = cfg['vr']*cfg['gamma']**(cfg['nb_states']-2)
                perf_stop = 0.5*p_star + 0.5*p_rand

            # random MDPs experiment
            elif cfg['environment'] == 'garnets':
                env = garnets.Garnets(cfg['nb_states'], cfg['nb_actions'], cfg['connectivity'], 0)
                _, _, p_star, _, p_rand, _ =env.generate_baseline_policy(cfg['gamma'], softmax_target_perf_ratio=-1,
                                                                                    baseline_target_perf_ratio=-1, farthest=cfg['farthest'])
                perf_stop=0.99*p_star + 0.01*p_rand

            prt(p_star)
            prt(p_rand)

            expe['env'] = env
            expe['p_star'] = p_star
            expe['p_rand'] = p_rand

            # creates the learning object:
            pa = gradient_ascent.GradientAscent(cfg['gamma'], cfg['nb_states'], cfg['nb_actions'], env, env.transition_function,
                                                env.reward_function, max_nb_it=cfg['max_nb_it'], perf_stop=perf_stop)

            if cfg['setting'] == 'sample':
                for algo in expe['cfg']['algos']:
                    algo['res'] = {}
                    if False:
                    # if algo['discounting'] == 'jh' or algo['discounting'] == 'jh2':
                        perf_glob, perf_jekyll = pa.fit_samples(algo['parametrization'], algo['discounting'], algo['critic_type'],
                                                                algo['actor_stepsize'], critic_stepsize=algo['critic_stepsize'],
                                                                init_q=algo['init_q'], hyde_param=algo['hyde_param'],
                                                                alpha=algo['alpha'], lambada=algo['lambada'], 
                                                                critic_UCB_stepsize=algo['critic_UCB_stepsize'],
                                                                minibatch_size=algo['minibatch_size'], 
                                                                offpol_param=algo['offpol_param'],
                                                                offpol_alpha=algo['offpol_alpha'])
                        algo['res']['perf_glob'] = perf_glob
                        algo['res']['perf_jekyll'] = perf_jekyll
                        
                    else:
                        algo['res']['perf_glob'] = pa.fit_samples(algo['parametrization'], algo['discounting'], algo['critic_type'],
                                                                    algo['actor_stepsize'], critic_stepsize=algo['critic_stepsize'],
                                                                    init_q=algo['init_q'], hyde_param=algo['hyde_param'],
                                                                    alpha=algo['alpha'], lambada=algo['lambada'], 
                                                                    critic_UCB_stepsize=algo['critic_UCB_stepsize'],
                                                                minibatch_size=algo['minibatch_size'], 
                                                                offpol_param=algo['offpol_param'],
                                                                offpol_alpha=algo['offpol_alpha']) 
            elif cfg['setting'] == 'exact':
                for algo in expe['cfg']['algos']:
                    algo['res'] = {}
                    algo['res']['perf_glob'] = pa.fit_exact(algo['parametrization'], algo['discounting'], algo['actor_stepsize'], 
                                                            hyde_param=algo['hyde_param'], alpha=algo['alpha'], lambada=algo['lambada']) 
            
            # Store data (serialize)
            pkl_file = config_file + '_' + str(expe['seed']) + '.pkl'
            with open(pkl_file, 'wb') as handle:
                pickle.dump(expe, handle, protocol=pickle.HIGHEST_PROTOCOL)
                prt('Experiment saved to ' + pkl_file)

config_file = 'expes/expe_name/' + 'config_name'
run_experiment(config_file) 