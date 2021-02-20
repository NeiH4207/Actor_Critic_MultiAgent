from __future__ import division
import random
from copy import deepcopy as dcopy

from src.environment import Environment
from src.agents import Agent
from read_input import Data
from itertools import count
import numpy as np
from src.utils import flatten
from collections import deque
import time
from src.utils import plot

def train(args, args_2): 
    data = Data(args.min_size, args.max_size)
    env = Environment(data.get_random_map(), args.show_screen, args.max_size)
    agent_1 = Agent(args, env.observation_dim, env.action_dim, 'agent_procon_1')  
    agent_2 = Agent(args_2, env.observation_dim, env.action_dim, 'agent_procon_2')  
    visual_mean_value_1 = deque(maxlen = 5000)
    visual_mean_value_2 = deque(maxlen = 5000)
    visual_mean_value_3 = deque(maxlen = 5000)
    visual_mean_value_4 = deque(maxlen = 5000)
    visual_value_1 = deque(maxlen = 10)
    visual_value_2 = deque(maxlen = 10)
    visual_value_3 = deque(maxlen = 100)
    visual_value_4 = deque(maxlen = 100)
    lr_super = args.lr_super
    lr_super_2 = args_2.lr_super
    
    cnt_w = 0
    cnt_l = 0
    
    for _ep in range(args.n_epochs):
        print('Training_epochs: {}'.format(_ep + 1))
        agent_1.set_environment(env.n_agents)
        agent_2.set_environment(env.n_agents)
        
        for _game in range(args.n_games):
            done = False
            start = time.time()
            for _iter in count():
                if args.show_screen:
                    env.render()
                # initialize
                states_1, actions_1, state_values_1, log_probs_1, rewards_1 = [], [], [], [], []
                states_2, actions_2, state_values_2, log_probs_2, rewards_2 = [], [], [], [], []
                # update by step
                soft_state_1 = env.get_observation(0)
                soft_agent_pos_1 = env.get_agent_pos(0)
                
                soft_state_2 = env.get_observation(1)
                soft_agent_pos_2 = env.get_agent_pos(1)
                
                predict_actions_1 = agent_1.select_action_smart(soft_state_1, soft_agent_pos_1, env)
                predict_actions_2 = agent_2.select_action_smart(soft_state_2, soft_agent_pos_2, env)
                
                # fit for each agent
                for agent_id in range(env.n_agents):
                    agent_state_1 = np.array(flatten([soft_state_1, env.get_agent_state(agent_id, soft_agent_pos_1)]), dtype = np.float32)
                    if random.random() < lr_super:
                        action_1, log_prob_1, state_value_1 = agent_1.select_action_exp(agent_state_1, predict_actions_1[agent_id])
                    else:
                        action_1, log_prob_1, state_value_1 = agent_1.select_action(agent_state_1)
                    state_values_1.append(state_value_1)
                    valid_1, next_state_1, reward_1 = env.soft_step(agent_id, soft_state_1, action_1, soft_agent_pos_1)
                    soft_state_1 = next_state_1
                    
                    states_1.append(agent_state_1)
                    actions_1.append(action_1)
                    log_probs_1.append(log_prob_1)
                    rewards_1.append(reward_1)
                    
                    # agent_state_2 = np.array(flatten([soft_state_2, env.get_agent_state(agent_id, soft_agent_pos_2)]))
                    # if random.random() < lr_super_2:
                    #     action_2, log_prob_2, state_value_2 = agent_2.select_action_exp(agent_state_2, predict_actions_2[agent_id])
                    # else:
                    #     action_2, log_prob_2, state_value_2 = agent_2.select_action(agent_state_2)
                        
                    # action_2, log_prob_2, state_value_2 = agent_2.select_action(agent_state_2)
                    # state_values_2.append(state_value_2)
                    # valid_2, next_state_2, reward_2 = env.soft_step(agent_id, soft_state_2, action_2, soft_agent_pos_2)
                    # soft_state_2 = next_state_2
                    # states_2.append(agent_state_2)
                    # actions_2.append(action_2)
                    # log_probs_2.append(log_prob_2)
                    # rewards_2.append(reward_2)
                    actions_2.append(np.random.randint(0, agent_2.action_dim - 1))
                
                # actions_2 = [0] * agent_.n_agents
                # actions_2 = agent.select_action_smart(soft_state_2, soft_agent_pos_2, env)
                # time.sleep(100000000)
                next_state, final_reward, done, _ = env.step(actions_1, actions_2, args.show_screen)
                for i in range(env.n_agents):
                    rewards_1[i] = 0
                    # rewards_2[i] = 0
                    if done:
                        if i == env.n_agents - 1:
                            rewards_1[i] = final_reward
                            # rewards_2[i] = - final_reward
                    agent_1.model.store(log_probs_1[i], state_values_1[i], rewards_1[i], next_state_1)
                    # agent_2.model.store(log_probs_2[i], state_values_2[i], rewards_2[i], next_state_2)
                if done:
                    if final_reward >= 0:
                        cnt_w += 1
                    else:
                        cnt_l += 1
                    break
                
            agent_1.learn()
            # agent_2.learn()
            end = time.time()
            visual_value_3.append(agent_1.value_loss)
            visual_value_4.append(agent_2.value_loss)
            visual_value_2.append(cnt_l)
            visual_value_1.append(cnt_w)
            visual_mean_value_1.append(np.mean(visual_value_1))
            visual_mean_value_2.append(np.mean(visual_value_2))
            visual_mean_value_3.append(np.mean(visual_value_3))
            visual_mean_value_4.append(np.mean(visual_value_4))
            if agent_1.steps_done % 50 == 0:
                plot(visual_mean_value_1, False, 'red')
                plot(visual_mean_value_2, True, 'blue')
                plot(visual_mean_value_3, False, 'red')
                plot(visual_mean_value_4, True, 'blue')
                print("Time: {0: >#.3f}s". format(1000*(end - start)))
            if args.saved_checkpoint:
                if _game % 50 == 0:
                    agent_1.save_models()
                    agent_2.save_models()
            env.soft_reset()
        # print('Completed episodes', lr_super)
        lr_super *= 0.985
        env.punish = 0
        # env = Environment(data.get_random_map(), args.show_screen, args.max_size)

"""
Created on Fri Nov 27 16:00:47 2020
@author: hien
"""
import argparse
from test_model1 import test_env

def get_args_1():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Procon""")
    parser.add_argument("--file_name", default = "input.txt")
    parser.add_argument("--type", default = "1")
    parser.add_argument("--run", type=str, default="train")   
    parser.add_argument("--min_size", type=int, default= 10)   
    parser.add_argument("--max_size", type=int, default= 20)   
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=128, help="The number of state per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_super", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--discount", type=float, default=0.999)   
    parser.add_argument("--ltv", type=float, default=1.1)   
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--replay_memory_size", type=int, default=10000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--n_games", type=int, default=10)
    parser.add_argument("--n_maps", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=1000000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--test_model", type=bool, default=False)
    parser.add_argument("--chkpoint", type=str, default='./Models/')
    parser.add_argument("--show_screen", type=str, default=True)
    parser.add_argument("--load_checkpoint", type=str, default=True)
    parser.add_argument("--saved_checkpoint", type=str, default=True)   
    
    args, unknown = parser.parse_known_args()
    return args

def get_args_2():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Procon""")
    parser.add_argument("--file_name", default = "input.txt")
    parser.add_argument("--type", default = "1")
    parser.add_argument("--run", type=str, default="train")   
    parser.add_argument("--min_size", type=int, default= 6)   
    parser.add_argument("--max_size", type=int, default= 6)   
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=128, help="The number of state per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_super", type=float, default=0.00)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--discount", type=float, default=0.999)   
    parser.add_argument("--ltv", type=float, default=1.1)   
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--replay_memory_size", type=int, default=10000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--n_games", type=int, default=10)
    parser.add_argument("--n_maps", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=1000000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--test_model", type=bool, default=False)
    parser.add_argument("--chkpoint", type=str, default='./Models/')
    parser.add_argument("--show_screen", type=str, default=True)
    parser.add_argument("--load_checkpoint", type=str, default=True)
    parser.add_argument("--saved_checkpoint", type=str, default=True)   
    
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args_1 = get_args_1()
    args_2 = get_args_2()
    if args_1.run == "train":
        train(args_1, args_2)
    if args_1.run == "test_model":
        test_env(args_1)