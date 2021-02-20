#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:45:45 2020

@author: hien
"""
import numpy as np
import torch
from src.model import ActorCritic, ActorCritic_2
from src.replay_memory import ReplayBuffer
from random import random, randint, choices, uniform
from src import utils
from src.utils import flatten
import torch.nn.functional as F
from sklearn.utils import shuffle
from copy import deepcopy as dcopy
from torch.distributions import Categorical

MAP_SIZE = 5

class Agent():
    def __init__(self, args, observation_dim, action_dim, agent_name):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.args = args
        self.iter = 0
        self.steps_done = 0
        self.nrand_action = 0
        self.learn_step_counter = 0
        self.random_rate = self.args.initial_epsilon
        self.agent_name = agent_name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.use_cuda = torch.cuda.is_available()
        self.chk_point_file_model = './Models/'
        self.value_loss = 0
        self.policy_loss = 0
        self.n_inputs = 8
        ''' Setup CUDA Environment'''
        self.device = 'cuda' if self.use_cuda else 'cpu'
        
        self.model = ActorCritic(self.n_inputs, self.observation_dim, self.action_dim, self.args.lr, self.chk_point_file_model)
        self.target_model = dcopy(self.model)
        self.model.to(self.device)
        if self.args.load_checkpoint:
            self.load_models()
        
        self.memories = ReplayBuffer(self.args.replay_memory_size)
            
            
    def set_environment(self, n_agents):
        self.n_agents = n_agents
        
    def convert_one_hot(self, action):
        n_values = self.action_dim
        return np.eye(n_values, dtype = np.float32)[action]
    
    def convert_one_hot_tensor(self, next_pred_action):
        return torch.zeros(len(next_pred_action), 
                           self.action_dim).scatter_(1, next_pred_action.unsqueeze(1), 1.)
    
    def record(self, state, action, log_prob, reward, next_state):
        action = self.convert_one_hot(action)
        self.memories.store_transition(state, action, log_prob, reward, next_state)

    def learn(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        
        log_probs, y_pred, rewards, next_states = self.model.get_data()
        
        ''' ---------------------- optimize ----------------------
        Use target actor exploitation policy here for loss evaluation
        y_exp = r + gamma*Q'( s2, pi'(s2))
        y_pred = Q( s1, a1)
        '''
        # gae = torch.zeros(1, 1).to
        for i in reversed(range(len(rewards) - 1)):
            r = rewards[i + 1]
            rewards[i] = rewards[i] + self.args.gamma * r
            # gae = gae * self.args.gamma + rewards[i] - y_pred[i] + \
            #     (y_pred[i + 1] if i < len(rewards) - 1 else 0)
        # print(rewards)
        y_exp = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        y_pred = torch.stack(y_pred).unsqueeze(1).to(self.device)
        log_probs = torch.stack(log_probs).unsqueeze(1).to(self.device)
        advantage = y_exp - y_pred
        policy_loss = (-log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss - 0.001 * self.model.entropies
        self.model.reset_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.model.optimize()
        self.model.clear()
        # for parameter in self.model.parameters():
            # print(parameter.grad)
        self.policy_loss = policy_loss.mean().to('cpu').data.numpy()
        self.value_loss = value_loss.mean().to('cpu').data.numpy()
        
        self.steps_done += 1
        

    def learn_batch_size(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        # print(self.memories.len)
        if self.memories.len < self.args.batch_size:
            return
        
        states, log_probs, rewards, next_states = self.memories.sample(self.args.batch_size)   
        
        ''' ---------------------- optimize ----------------------
        Use target actor exploitation policy here for loss evaluation
        y_exp = r + gamma*Q'( s2, pi'(s2))
        y_pred = Q( s1, a1)
        '''
        _, next_qval = self.target_model.forward(next_states)
        _, y_pred = self.target_model.forward(states)
        y_exp = rewards + self.args.gamma * next_qval.detach()
        
        policy_loss = []
        value_loss = []
        
        # y_exp = (y_exp - y_exp.mean()) / (y_exp.std() + np.finfo(np.float32).eps.item()) 
        for i in range(len(rewards)):
            log_prob, exp_reward, pred_reward = log_probs[i], y_exp[i], y_pred[i]  
            delta = exp_reward.item() - pred_reward.item()
            policy_loss.append(-log_prob * delta)
            value_loss.append(F.smooth_l1_loss(pred_reward, torch.tensor([exp_reward]).to(self.device)))
            # print([pred_reward.item(), exp_reward.item()])
        policy_loss = torch.stack(policy_loss)
        value_loss = torch.stack(value_loss)
        # print(policy_loss, value_loss)
        self.model.reset_grad()
        (policy_loss.sum() + value_loss.sum()).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.model.optimize()
        # for parameter in self.model.parameters():
            # print(parameter.grad)
        self.model.clear()
        self.policy_loss = policy_loss.mean().to('cpu').data.numpy()
        self.value_loss = value_loss.mean().to('cpu').data.numpy()
        
        self.steps_done += 1
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        prob, state_value = self.model(state)
        act = prob.sample()
        if random() < self.random_rate:
            act = torch.tensor(randint(0, self.action_dim - 1)).to(self.device)
        log_p = prob.log_prob(act)
        self.model.entropies += prob.entropy().mean()
        act = int(act.to('cpu').numpy())
        return act, log_p, state_value
    
    
    def select_action_exp(self, state, action):
        state = torch.FloatTensor(state).to(self.device)
        prob, state_value = self.model(state)
        log_p = prob.log_prob(torch.tensor(action).to(self.device))
        self.model.entropies += prob.entropy().mean()
        return action, log_p, state_value
                      
        
    def select_action_smart(self, state, agent_pos, env):
        score_matrix, agents_matrix, conquer_matrix, \
                       treasures_matrix, walls_matrix = [dcopy(_) for _ in state]
        actions = [0] * self.n_agents
        state = dcopy(state)
        agent_pos = dcopy(agent_pos)
        init_score = 0
        order = shuffle(range(self.n_agents))
        
        for i in range(self.n_agents):
            agent_id = order[i]
            
            act = 0
            scores = [0] * 9
            mn = 1000
            mx = -1000
            valid_states = []
            for act in range(9):
                _state, _agent_pos = dcopy([state, agent_pos])
                valid, next_state, reward = env.soft_step(agent_id, _state, act, _agent_pos, predict=True)
                scores[act] = reward - init_score
                mn = min(mn, reward - init_score)
                mx = max(mx, reward - init_score)
                valid_states.append(valid)
                
            scores[0] = mn
            for j in range(len(scores)):
                scores[j] = (scores[j] - mn) / (mx - mn + 0.0001)
    
            sum = np.sum(scores) + 0.0001
            for j in range(len(scores)):
                scores[j] = scores[j] / sum
                if(valid_states[j] is False):
                    scores[j] = 0
            act = np.array(scores).argmax()
            valid, state, score = env.soft_step(agent_id, state, act, agent_pos, predict=True)
            init_score = score
            actions[agent_id] = act
            
        return actions
    
    def select_action_test_not_predict(self, state):
        actions = []
        state = dcopy(state)
        state = np.reshape(flatten(state), (7, 20, 20))
        state = [state[0], [state[1], state[2]], [state[3], state[4]], state[5], state[6]]
        agent_pos_1 = dcopy(self.env.agent_pos_1)
        agent_pos_2 = dcopy(self.env.agent_pos_2)
        init_score = self.env.score_mine - self.env.score_opponent
        rewards = []
        states = []
        next_states = []
        
        for i in range(self.num_agents):
            _state = state
            _state[1] = self.env.get_agent_state(_state[1], i)
            _state = flatten(_state)
            states.append(state)
            act = 0
            scores = [0] * 9
            mn = 1000
            mx = -1000
            valid_states = []
            for act in range(9):
                _state, _agent_pos_1, _agent_pos_2 = dcopy([state, agent_pos_1, agent_pos_2])
                valid, _state, _agent_pos, _score = self.env.fit_action(i, _state, act, _agent_pos_1, _agent_pos_2, False)
                scores[act] = _score - init_score
                mn = min(mn, _score - init_score)
                mx = max(mx, _score - init_score)
                valid_states.append(valid)
            for j in range(len(scores)):
                scores[j] = (scores[j] - mn) / (mx - mn + 0.0001)
                scores[j] **= 5
            sum = np.sum(scores) + 0.0001
            for j in range(len(scores)):
                scores[j] = scores[j] / sum
                if(valid_states[j] is False):
                    scores[j] = 0
            act = choices(range(9), scores)[0]
            valid, state, agent_pos, score = self.env.fit_action(i, state, act, agent_pos_1, agent_pos_2)
            init_score = score
            actions.append(act)
            next_states.append(state)
            
        return states, actions, rewards, next_states
    
    def select_random(self, state):
        actions = []
        for i in range(self.num_agents):
            actions.append(randint(0, 8))
        return state, actions, [0] * self.num_agents, state 
    
    def save_models(self):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:        
        """
        self.model.save_checkpoint(self.agent_name)
        
    def load_models(self):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.model.load_checkpoint(self.agent_name)
        
        self.model.eval()