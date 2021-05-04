#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 07:59:17 2021

@author: hien
"""
from __future__ import division
import random
from src.environment import Environment
from src.agents import Agent
from read_input import Data
from itertools import count
import numpy as np
from collections import deque
from sklearn.utils import shuffle
import time
import torch
from src.utils import plot, dotdict

cargs = dotdict({
    'run_mode': 'test',
    'visualize': True,
    'min_size': 6,
    'max_size': 6,
    'n_games': 1,
    'num_iters': 20000,
    'n_epochs': 1000000,
    'n_maps': 1000,
    'show_screen': True,
})

args = [
        dotdict({
            'optimizer': 'adas',
            'lr': 1e-4,
            'exp_rate': 0.0,
            'gamma': 0.99,
            'tau': 0.01,
            'max_grad_norm': 0.3,
            'discount': 0.6,
            'num_channels': 64,
            'batch_size': 256,
            'replay_memory_size': 100000,
            'dropout': 0.6,
            'initial_epsilon': 0.0,
            'final_epsilon': 1e-4,
            'load_folder_file': ('Models','agent_mcts.pt'),
            'load_checkpoint': True,
            'saved_checkpoint': True
        }),
        
        dotdict({
            'optimizer': 'adas',
            'lr': 1e-4,
            'exp_rate': 0.7,
            'gamma': 0.99,
            'tau': 0.01,
            'max_grad_norm': 0.3,
            'discount': 0.6,
            'batch_size': 256,
            'num_channels': 64,
            'replay_memory_size': 100000,
            'dropout': 0.4,
            'initial_epsilon': 0.0,
            'final_epsilon': 0.01,
            'load_folder_file': ('Models','agent_1.pt'),
            'load_checkpoint': False,
            'saved_checkpoint': True
        })
]
        
def test(): 
    data = Data(cargs.min_size, cargs.max_size)
    env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)
    agent = [Agent(env, args[0]), Agent(env, args[1])]
    wl_mean, score_mean = [[deque(maxlen = 10000), deque(maxlen = 10000)]  for _ in range(2)]
    wl, score = [[deque(maxlen = 1000), deque(maxlen = 1000)] for _ in range(2)]
    cnt_w, cnt_l = 0, 0
    exp_rate = [args[0].exp_rate, args[1].exp_rate]
    # agent[0].model.load_state_dict(torch.load(checkpoint_path_1, map_location = agent[0].model.device))
    # agent[1].model.load_state_dict(torch.load(checkpoint_path_2, map_location = agent[1].model.device))
        
    for _ep in range(cargs.n_epochs):
        if _ep % 10 == 9:
            print('Testing_epochs: {}'.format(_ep + 1))
        done = False
        start = time.time()
        current_state = env.get_observation(0)
        for _iter in range(env.n_turns):
            if cargs.show_screen:
                env.render()
                
            """ initialize """
            actions, soft_state, soft_agent_pos, pred_acts, exp_rewards = \
                [[[], []] for i in range(5)]
                
            """ update by step """
            for i in range(env.num_players):
                soft_state[i] = env.get_observation(i)
                soft_agent_pos[i] = env.get_agent_pos(i)
                pred_acts[i], exp_rewards[i] = agent[i].select_action_smart(soft_state[i], soft_agent_pos[i], env)

            """ select action for each agent """
            for agent_id in range(env.n_agents):
                for i in range(env.num_players):
                    ''' get state to forward '''
                    state_step = env.get_states_for_step(current_state)
                    agent_step = env.get_agent_for_step(agent_id, i, soft_agent_pos)
                    ''' predict from model'''
                    if random.random() < exp_rate[i]:
                        act = pred_acts[i][agent_id]
                    else:
                        # print(i)
                        act = agent[i].get_action(state_step, agent_step)
                        # act, _, _ = agent[i].select_action(state_step, agent_step)
                    ''' convert state to opponent state '''
                    env.convert_to_opn_obs(current_state, soft_agent_pos)
                    ''' storage infomation trainning'''
                    actions[i].append(act)
                ''' last action to fit next state '''
                acts = [actions[0][-1], actions[1][-1]]
                current_state, temp_rewards = env.soft_step_2(agent_id, current_state, acts, soft_agent_pos)
                    
            # actions[1] = [np.random.randint(0, env.n_actions - 1) for _ in range(env.n_agents)]
            # actions[1] = [0] * env.n_agents
            # actions[1] = pred_acts[1]
            current_state, final_reward, done, _ = env.step(actions[0], actions[1], cargs.show_screen)
            if done:
                score[0].append(env.players[0].total_score)
                score[1].append(env.players[1].total_score)
                if env.players[0].total_score > env.players[1].total_score:
                    cnt_w += 1
                else:
                    cnt_l += 1
                break
            
        end = time.time()
            
        wl[0].append(cnt_w)
        wl[1].append(cnt_l)
        for i in range(2):
            wl_mean[i].append(np.mean(wl[i]))
            score_mean[i].append(np.mean(score[i]))
                
        if _ep % 50 == 49:
            plot(wl_mean, vtype = 'Win')
            plot(score_mean, vtype = 'Score')
            print("Time: {0: >#.3f}s". format(1000*(end - start)))
        env.soft_reset()
        # env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)

if __name__ == "__main__":
    test()