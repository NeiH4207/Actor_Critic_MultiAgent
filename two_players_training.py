from __future__ import division
import random
from src.environment import Environment
from src.agents import Agent
from read_input import Data
from itertools import count
import numpy as np
from collections import deque
import time
import torch
from src.utils import plot, dotdict

cargs = dotdict({
    'run_mode': 'train',
    'visualize': True,
    'min_size': 7,
    'max_size': 7,
    'n_games': 3,
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
            'initial_epsilon': 0.1,
            'final_epsilon': 1e-4,
            'dir': './Models/',
            'load_checkpoint': False,
            'saved_checkpoint': True
        }),
        
        dotdict({
            'optimizer': 'adas',
            'lr': 1e-4,
            'exp_rate': 0.0,
            'gamma': 0.99,
            'tau': 0.01,
            'max_grad_norm': 0.3,
            'discount': 0.6,
            'batch_size': 256,
            'num_channels': 64,
            'replay_memory_size': 100000,
            'dropout': 0.4,
            'initial_epsilon': 0.1,
            'final_epsilon': 0.01,
            'dir': './Models/',
            'load_checkpoint': False,
            'saved_checkpoint': True
        })
]

def train(): 
    data = Data(cargs.min_size, cargs.max_size)
    env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)
    agent = [Agent(env, args[0], 'agent_1'), Agent(env, args[1], 'agent_2')]
    wl_mean, score_mean, l_val_mean =\
        [[deque(maxlen = 10000), deque(maxlen = 10000)]  for _ in range(3)]
    wl, score, l_val = [[deque(maxlen = 1000), deque(maxlen = 1000)] for _ in range(3)]
    lr_super = [args[0].exp_rate, args[1].exp_rate]
    cnt_w, cnt_l = 0, 0
    # agent[0].model.load_state_dict(torch.load(checkpoint_path_1, map_location = agent[0].model.device))
    # agent[1].model.load_state_dict(torch.load(checkpoint_path_2, map_location = agent[1].model.device))
        
    for _ep in range(cargs.n_epochs):
        if _ep % 10 == 9:
            print('Training_epochs: {}'.format(_ep + 1))
        for _game in range(cargs.n_games):
            done = False
            start = time.time()
            for _iter in count():
                if cargs.show_screen:
                    env.render()
                    
                """ initialize """
                actions, state_vals, log_probs, rewards, soft_state, \
                    soft_agent_pos, pred_acts, exp_rewards = [[[], []] for i in range(8)]
                    
                """ update by step """
                for i in range(env.num_players):
                    soft_state[i] = env.get_observation(i)
                    soft_agent_pos[i] = env.get_agent_pos(i)
                    pred_acts[i], exp_rewards[i] = agent[i].select_action_smart(soft_state[i], soft_agent_pos[i], env)

                """ select action for each agent """
                for agent_id in range(env.n_agents):
                    for i in range(env.num_players):
                        agent_state = env.get_states_for_step(soft_state[i])
                        # not change
                        agent_step = env.get_agent_for_step(agent_id, soft_agent_pos)
                        act, log_p, state_val = 0, 0, 0
                        if random.random() < lr_super[i]:
                            act, log_p, state_val = agent[i].select_action_by_exp(
                                agent_state, agent_step, pred_acts[i][agent_id])
                        else:
                            act, log_p, state_val = agent[i].select_action(agent_state, agent_step)
                                
                        soft_state[i] = env.soft_step_(agent_id, soft_state[i], act, soft_agent_pos[i])
                        state_vals[i].append(state_val)
                        actions[i].append(act)
                        log_probs[i].append(log_p)
                # actions[1] = [np.random.randint(0, env.n_actions - 1) for _ in range(env.n_agents)]
                # actions[1] = [0] * env.n_agents
                # actions[1] = pred_acts[1]
                next_state, final_reward, done, _ = env.step(actions[0], actions[1], cargs.show_screen)
                for i in range(env.n_agents):
                    rewards[0].append(final_reward)
                    rewards[1].append(- final_reward)
                    for j in range(env.num_players):
                        if pred_acts[j][i] == actions[j][i]:
                            reward = exp_rewards[j][i]
                            beta = 0.9
                            rewards[j][i] = rewards[j][i] * (1 - beta)  + beta * reward
                        agent[j].model.store(log_probs[j][i], state_vals[j][i], rewards[j][i])
                if done:
                    score[0].append(env.players[0].total_score)
                    score[1].append(env.players[1].total_score)
                    if env.players[0].total_score > env.players[1].total_score:
                        cnt_w += 1
                    else:
                        cnt_l += 1
                    break
            agent[0].learn()
            agent[1].learn()
            end = time.time()
            if _ep > 3:
                l_val[0].append(agent[0].value_loss)
                l_val[1].append(agent[1].value_loss)
                wl[0].append(cnt_w)
                wl[1].append(cnt_l)
                for i in range(2):
                    wl_mean[i].append(np.mean(wl[i]))
                    score_mean[i].append(np.mean(score[i]))
                    l_val_mean[i].append(np.mean(l_val[i]))
            
            env.soft_reset()
        if _ep % 50 == 49:
            if cargs.visualize:
                plot(wl_mean, vtype = 'Win')
                plot(score_mean, vtype = 'Score')
                plot(l_val_mean, vtype = 'Loss_Value')
                print("Time: {0: >#.3f}s". format(1000*(end - start)))
            if args[0].saved_checkpoint:
                agent[0].save_models()
                # torch.save(agent[0].model.state_dict(), checkpoint_path_1)
            if args[1].saved_checkpoint:
                agent[1].save_models()
                # torch.save(agent[1].model.state_dict(), checkpoint_path_2)
                # print('Completed episodes')
        # lr_super *= 0.999
        env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)
        
def test(): 
    data = Data(cargs.min_size, cargs.max_size)
    env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)
    agent = [Agent(env, args[0], 'agent_1'), Agent(env, args[1], 'agent_2')]
    wl_mean, score_mean = [[deque(maxlen = 10000), deque(maxlen = 10000)]  for _ in range(2)]
    wl, score = [[deque(maxlen = 1000), deque(maxlen = 1000)] for _ in range(2)]
    cnt_w, cnt_l = 0, 0
    # agent[0].model.load_state_dict(torch.load(checkpoint_path_1, map_location = agent[0].model.device))
    # agent[1].model.load_state_dict(torch.load(checkpoint_path_2, map_location = agent[1].model.device))
        
    for _ep in range(cargs.n_epochs):
        if _ep % 10 == 9:
            print('Testing_epochs: {}'.format(_ep + 1))
        done = False
        start = time.time()
        for _iter in count():
            if cargs.show_screen:
                env.render()
                
            """ initialize """
            actions, soft_state, soft_agent_pos = [[[], []] for i in range(3)]
                
            """ update by step """
            for i in range(env.num_players):
                soft_state[i] = env.get_observation(i)
                soft_agent_pos[i] = env.get_agent_pos(i)
               
            """ select action for each agent """
            for agent_id in range(env.n_agents):
                for i in range(env.num_players):
                    agent_state = env.get_states_for_step(soft_state[i])
                    agent_step = env.get_agent_for_step(agent_id, soft_agent_pos[i])
                    act, log_p, state_val = agent[i].select_action(agent_state, agent_step)
                            
                    soft_state[i] = env.soft_step_(agent_id, soft_state[i], act, soft_agent_pos[i])
                    actions[i].append(act)
            # actions[1] = [np.random.randint(0, env.n_actions - 1) for _ in range(env.n_agents)]
            # actions[1] = [0] * env.n_agents
            # actions[1] = pred_acts[1]
            next_state, final_reward, done, _ = env.step(actions[0], actions[1], cargs.show_screen)
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
        env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)

"""
Created on Fri Nov 27 16:00:47 2020
@author: hien
"""
if __name__ == "__main__":
        # lr_super *= 0.999
        # lr_super *= 0.999
    if cargs.run_mode == "train":
        train()
    if cargs.run_mode == "test":
        test()