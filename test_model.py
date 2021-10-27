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
from tqdm import tqdm

cargs = dotdict({
    'visualize': False,
    'show_screen': True,
    'min_size': 7,
    'max_size': 7,
    'n_games': 30,
    'n_tests': 5,
})

args = [
        dotdict({
            'exp_rate': 0.0,
            'run_mode': 'test',
            'load_folder_file': ('Models','agent_mcts.pt'),
            'colab_train': False,
            'colab_dir': "/content/drive/MyDrive/trainned_model/agent_mcts.pt",
            'load_checkpoint': True,
            'saved_checkpoint': False,
        }),
        
        dotdict({
            'exp_rate': 0.0,
            'run_mode': 'test',
            'load_folder_file': ('Models','agent_1.pt'),
            'colab_train': False,
            'colab_dir': "/content/drive/MyDrive/trainned_model/agent_mcts.pt",
            'load_checkpoint': False,
            'saved_checkpoint': False
        })
]
        
def test(): 
    data = Data(cargs.min_size, cargs.max_size)
    env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)
    agent = [Agent(env, args[0]), Agent(env, args[1])]
    
    if args[0].colab_train:
        agent[0].model.load_state_dict(torch.load(args[0].colab_dir, map_location = agent[0].model.device))
    if args[1].colab_train:
        agent[1].model.load_state_dict(torch.load(args[1].colab_dir, map_location = agent[1].model.device))
        
    agent[0].model.load_checkpoint(args[0].load_folder_file[0], args[0].load_folder_file[1])
    
    # agent[1].model.load_state_dict(torch.load(checkpoint_path_2, map_location = agent[1].model.device))
    
    for _t in range(cargs.n_tests):
        exp_rate = [args[0].exp_rate, _t / cargs.n_tests]
        n_wins = 0
        start = time.time()
        for game in tqdm(range(cargs.n_games), desc="Test {}".format(_t + 1)): 
            done = False
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
                    for player_id in range(env.num_players):
                        ''' get state to forward '''
                        state_step = env.get_states_for_step(current_state)
                        agent_step = env.get_agent_for_step(agent_id, i, soft_agent_pos)
                        ''' predict from model'''
                        if random.random() < exp_rate[player_id]:
                            act = pred_acts[player_id][agent_id]
                        else:
                            act = agent[player_id].get_action(state_step, agent_step, current_state, soft_agent_pos[player_id][agent_id])
                            # act, _, _ = agent[player_id].select_action(state_step, agent_step)
                        ''' convert state to opponent state '''
                        env.convert_to_opn_obs(current_state, soft_agent_pos)
                        ''' storage infomation trainning'''
                        actions[player_id].append(act)
                    ''' last action to fit next state '''
                    acts = [actions[0][-1], actions[1][-1]]
                    current_state, temp_rewards = env.soft_step_2(agent_id, current_state, acts, soft_agent_pos)
                        
                # actions[1] = [np.random.randint(0, env.n_actions - 1) for _ in range(env.n_agents)]
                # actions[1] = [0] * env.n_agents
                # actions[1] = pred_acts[1]
                current_state, final_reward, done, _ = env.step(actions[0], actions[1], cargs.show_screen)
                if done:
                    if env.players[0].total_score > env.players[1].total_score:
                        n_wins += 1
                    break
                
            end = time.time()
                
            env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)
        print("Win rate:", n_wins / cargs.n_games)
        print("Time: {0: >#.3f}s". format(1000*(end - start)))

if __name__ == "__main__":
    test()