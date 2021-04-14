#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:45:45 2020

@author: hien
"""
import numpy as np
import torch
from src.model import Policy
from src.replay_memory import ReplayMemory
from random import random, randint, choices, uniform
from src.utils import flatten
import torch.nn.functional as F
from sklearn.utils import shuffle
from copy import deepcopy as dcopy
from torch.distributions import Categorical
torch.manual_seed(1)
MAP_SIZE = 5

class Agent():
    def __init__(self, env, args):
        self.args = args
        self.iter = 0
        self.steps_done = 0
        self.n_actions = env.action_dim
        self.learn_step_counter = 0
        self.random_rate = self.args.initial_epsilon
        self.use_cuda = torch.cuda.is_available()
        self.chk_point_file_model = './Models/'
        self.value_loss = 0
        self.policy_loss = 0
        self.n_inputs = 8
        ''' Setup CUDA Environment'''
        self.device = 'cuda' if self.use_cuda else 'cpu'
        
        self.model = Policy(env)
        self.model.to(self.device)
        if self.args.load_checkpoint:
            self.load_models()
        
        self.memories = ReplayMemory(self.args.replay_memory_size, self.args.batch_size)
            
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
        
        log_probs, y_pred, rewards = self.model.get_data()
        
        ''' ---------------------- optimize ----------------------
        Use target actor exploitation policy here for loss evaluation
        y_exp = r + gamma*Q'( s2, pi'(s2))
        y_pred = Q( s1, a1)
        '''
        # print(rewards)
        # gae = torch.zeros(1, 1).to
        for i in reversed(range(len(rewards[0]) - 1)):
            # rewards[i] = rewards[i] * self.args.discount + (1 - self.args.discount) * rewards[i + 1]
            rewards[0][i] = rewards[0][i] + self.args.gamma * rewards[0][i + 1]
            rewards[1][i] = rewards[1][i] + self.args.gamma * rewards[1][i + 1]
            # gae = gae * self.args.gamma + rewards[i] - y_pred[i] + \
            #     (y_pred[i + 1] if i < len(rewards) - 1 else 0)
        # print(rewards)
        
        y_exp = torch.cat((torch.FloatTensor(rewards[0]).unsqueeze(1).to(self.device),
                          torch.FloatTensor(rewards[1]).unsqueeze(1).to(self.device)),
                          dim = 1).view(-1)
        y_pred = torch.cat((torch.stack(y_pred[0]).unsqueeze(1).to(self.device),
                          torch.stack(y_pred[1]).unsqueeze(1).to(self.device)),
                          dim = 1).view(-1)
        log_probs = torch.cat((torch.stack(log_probs[0]).unsqueeze(1).to(self.device),
                          torch.stack(log_probs[1]).unsqueeze(1).to(self.device)),
                          dim = 1).view(-1)
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
        
    def select_action(self, state, agent):
        state = torch.FloatTensor(state).to(self.device)
        agent = torch.FloatTensor(agent).to(self.device)
        prob, state_value = self.model(state, agent)
        prob = torch.exp(prob)
        prob = Categorical(prob)
        act = prob.sample()
        if random() < self.random_rate:
            act = torch.tensor([randint(0, self.n_actions - 1)]).to(self.device)
            self.random_rate *= 0.9999
            self.random_rate = max(self.random_rate, self.args.final_epsilon)
        log_p = prob.log_prob(act)
        self.model.entropies += prob.entropy().mean()
        act = int(act.to('cpu').numpy())
        return act, log_p, state_value
    
    def get_action(self, state, agent):
        state = torch.FloatTensor(state).to(self.device)
        agent = torch.FloatTensor(agent).to(self.device)
        prob, state_value = self.model(state, agent)
        prob = torch.exp(prob).detach().to('cpu').numpy()[0]
        act = np.argmax(prob)
        return act
    
    def select_action_by_exp(self, state, agent, action):
        state = torch.FloatTensor(state).to(self.device)
        agent = torch.FloatTensor(agent).to(self.device)
        _, prob, state_value = self.model(state, agent)
        prob = Categorical(prob)
        log_p = prob.log_prob(torch.tensor(action).to(self.device))
        self.model.entropies += prob.entropy().mean()
        return action, log_p, state_value
                      
        
    def select_action_smart(self, state, agent_pos, env):
        score_matrix, agents_matrix, conquer_matrix, \
                       treasures_matrix, walls_matrix = [dcopy(_) for _ in state]
        actions = [0] * env.n_agents
        state = dcopy(state)
        agent_pos = dcopy(agent_pos)
        init_score = 0
        order = shuffle(range(env.n_agents))
        exp_rewards = [0] * env.n_agents
        
        for i in range(env.n_agents):
            agent_id = order[i]
            
            act = 0
            scores = [0] * 8
            mn = 1000
            mx = -1000
            valid_states = []
            for act in range(env.n_actions):
                _state, _agent_pos = dcopy([state, agent_pos])
                valid, next_state, reward = env.soft_step(agent_id, _state, act, _agent_pos, exp=True)
                scores[act] = reward - init_score
                mn = min(mn, reward - init_score)
                mx = max(mx, reward - init_score)
                valid_states.append(valid)
            
            # _scores = dcopy(scores)
            # scores[0] = mn
            # for j in range(len(scores)):
            #     scores[j] = (scores[j] - mn) / (mx - mn + 0.0001)
    
            # sum = np.sum(scores) + 0.0001
            # for j in range(len(scores)):
            #     scores[j] = scores[j] / sum
            #     if valid_states[j] is False:
            #         scores[j] = 0
            act = np.array(scores).argmax()
            valid, state, score = env.soft_step(agent_id, state, act, agent_pos, exp=True)
            init_score = score
            actions[agent_id] = act
            exp_rewards[agent_id] = mx
        return actions, exp_rewards
    
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
            scores = [0] * 8
            mn = 1000
            mx = -1000
            valid_states = []
            for act in range(8):
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
                if valid_states[j] is False:
                    scores[j] = 0
            act = choices(range(self.env.n_actions), scores)[0]
            valid, state, agent_pos, score = self.env.fit_action(i, state, act, agent_pos_1, agent_pos_2)
            init_score = score
            actions.append(act)
            next_states.append(state)
            
        return states, actions, rewards, next_states
    
    def select_random(self, state):
        actions = []
        for i in range(self.num_agents):
            actions.append(randint(0, 7))
        return state, actions, [0] * self.num_agents, state 
    
    def save_models(self):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:        
        """
        self.model.save_checkpoint()
        
    def load_models(self):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.model.load_checkpoint(self.args.load_folder_file[0],
                                   self.args.load_folder_file[1])
        
        self.model.eval()