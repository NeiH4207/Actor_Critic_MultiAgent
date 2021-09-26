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

args = dotdict({
    'run_mode': 'train',
    'visualize': True,
    'min_size': 10,
    'max_size': 10,
    'n_games': 1,
    'num_iters': 20000,
    'n_epochs': 1000000,
    'n_maps': 1000,
    'optimizer': 'adas',
    'lr': 1e-4,
    'exp_rate': 0.0,
    'gamma': 0.99,
    'tau': 0.01,
    'max_grad_norm': 0.3,
    'discount': 0.6,
    'batch_size': 256,
    'replay_memory_size': 100000,
    'initial_epsilon': 0.1,
    'final_epsilon': 1e-4,
    'dir': './Models/',
    'load_folder_file': ('Models','model.pt'),  
    'show_screen': False,
    'load_checkpoint': False,
    'saved_checkpoint': True
})

def train(): 
    data = Data(args.min_size, args.max_size)
    env = Environment(data.get_random_map(), args.show_screen, args.max_size)
    bot = Agent(env, args)
    
    wl_mean, score_mean, l_val_mean, l_pi_mean =\
        [[deque(maxlen = 10000), deque(maxlen = 10000)]  for _ in range(4)]
    wl, score, l_val, l_pi = [[deque(maxlen = 1000), deque(maxlen = 1000)] 
                                for _ in range(4)]
    cnt_w, cnt_l = 0, 0
    # bot.model.load_state_dict(torch.load(checkpoint_path_1, map_location = bot.model.device))
    # agent[1].model.load_state_dict(torch.load(checkpoint_path_2, map_location = agent[1].model.device))
        
    for _ep in range(args.n_epochs):
        if _ep % 10 == 9:
            print('Training_epochs: {}'.format(_ep + 1))
        for _game in range(args.n_games):
            done = False
            start = time.time()
            current_state = env.get_observation(0)
            
            for _iter in range(env.n_turns):
                if args.show_screen:
                    env.render()
                    
                """ initialize """
                actions, state_vals, log_probs, rewards, soft_agent_pos = [[[], []] for i in range(5)]
                    
                """ update by step """
                for i in range(env.num_players):
                    soft_agent_pos[i] = env.get_agent_pos(i)
                    
                """ select action for each agent """
                for agent_id in range(env.n_agents):
                    for i in range(env.num_players):
                        ''' get state to forward '''
                        state_step = env.get_states_for_step(current_state)
                        agent_step = env.get_agent_for_step(agent_id, soft_agent_pos)
                        ''' predict from model'''
                        act, log_p, state_val = bot.select_action(state_step, agent_step)
                        ''' convert state to opponent state '''
                        env.convert_to_opn_obs(current_state, soft_agent_pos)
                        ''' storage infomation trainning'''
                        state_vals[i].append(state_val)
                        actions[i].append(act)
                        log_probs[i].append(log_p)
                    ''' last action to fit next state '''
                    acts = [actions[0][-1], actions[1][-1]]
                    current_state, temp_rewards = env.soft_step_2(agent_id, current_state, acts, soft_agent_pos)
                    rewards[0].append(temp_rewards[0] - temp_rewards[1])
                    rewards[1].append(temp_rewards[1] - temp_rewards[0])
                    
                current_state, final_reward, done, _ = env.step(actions[0], actions[1], args.show_screen)
                for i in range(env.n_agents):
                    for j in range(env.num_players):
                        bot.model.store(j, log_probs[j][i], state_vals[j][i], rewards[j][i])
                        
            # store the win lose battle
            in_win = env.players[0].total_score > env.players[1].total_score
            if in_win: cnt_w += 1 
            else: cnt_l += 1
                
            score[0].append(env.players[0].total_score)
            score[1].append(env.players[1].total_score)
            bot.learn()
            end = time.time()
            if _ep > 3:
                l_val[0].append(bot.value_loss)
                l_pi[0].append(bot.policy_loss)
                # wl[0].append(cnt_w)
                # wl[1].append(cnt_l)
                for i in range(2):
                    # wl_mean[i].append(np.mean(wl[i]))
                    score_mean[i].append(np.mean(score[i]))
                    l_val_mean[i].append(np.mean(l_val[i]))
                    l_pi_mean[i].append(np.mean(l_pi[i]))
            
            env.soft_reset()
        if _ep % 100 == 99:
            if args.visualize:
                # plot(wl_mean, vtype = 'Win')
                plot(score_mean, vtype = 'ScoreTrue')
                plot(l_val_mean, vtype = 'Loss_Value')
                plot(l_pi_mean, vtype = 'Loss_Policy')
                print("Time: {0: >#.3f}s". format(1000*(end - start)))
            if args.saved_checkpoint:
                bot.save_models() 
                # torch.save(bot.model.state_dict(), checkpoint_path_1)
                # print('Completed episodes')
        env = Environment(data.get_random_map(), args.show_screen, args.max_size)
        
if __name__ == "__main__":
     train()