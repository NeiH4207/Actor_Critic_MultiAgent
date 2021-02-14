import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical
from AdasOptimizer.adasopt_pytorch import Adas
from torch.optim import Adam, SGD
from collections import deque

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, lr = 0.001, checkpoint_file = 'Models/model.pt', is_recurrent=True):
        super(ActorCritic, self).__init__()
        self.recurrent = is_recurrent
        self.action_dim = action_dim

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor_fc1 = nn.Linear(state_dim, 512)
        self.critic_fc1 = nn.Linear(state_dim, 512)
        self.actor_fc1.weight.data = fanin_init(self.actor_fc1.weight.data.size())
        self.critic_fc1.weight.data = fanin_init(self.critic_fc1.weight.data.size())
        
        self.actor_fc2 = nn.Linear(512, 128)
        self.critic_fc2 = nn.Linear(512, 128)
        self.actor_fc2.weight.data = fanin_init(self.actor_fc2.weight.data.size())
        self.critic_fc2.weight.data = fanin_init(self.critic_fc2.weight.data.size())
        self.lr = lr
        self.action_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1) # Scalar Value
        self.action_head.weight.data.uniform_(-EPS,EPS)
        self.value_head.weight.data.uniform_(-EPS,EPS)
        self.optimizer = Adas(self.parameters(), lr=self.lr)
        
        self.entropies = 0
        self.action_probs = []
        self.state_values = []
        self.rewards = []
        self.next_states = []
        self.checkpoint_file = checkpoint_file
    
    def store(self, prob, state_value, reward, next_state):
        self.action_probs.append(prob)
        self.state_values.append(state_value)
        self.rewards.append(reward)
        self.next_states.append(next_state)
    
    def clear(self):
        self.action_probs = []
        self.state_values = []
        self.rewards = []
        self.next_states = []
        self.entropies = 0
    
    def get_data(self):
        return self.action_probs, self.state_values, self.rewards, self.next_states
        
    def forward(self, state):
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        # x = F.relu(self.actor_fc3(x))
        action_score = self.action_head(x)
        
        y = F.relu(self.critic_fc1(state))
        y = F.relu(self.actor_fc2(y))
        # y = F.relu(self.actor_fc3(y))
        state_value = self.value_head(y)
        
        probs = F.softmax(action_score, dim = -1)
        return Categorical(probs), state_value

    def optimize(self):
        self.optimizer.step()
        
    def reset_grad(self):
        self.optimizer.zero_grad()

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(self.checkpoint_file, map_location = self.device))
        
        
    
class ActorCritic_2(nn.Module):
    def __init__(self, state_dim, action_dim, lr = 0.001, checkpoint_file = 'Models/model.pt'):
        super(ActorCritic_2, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor_fc1 = nn.Linear(state_dim, 64)
        self.critic_fc1 = nn.Linear(state_dim, 64)
        
        self.actor_fc2 = nn.Linear(64, 40)
        self.critic_fc2 = nn.Linear(64, 40)
        
        self.actor_fc3 = nn.Linear(32, 16)
        self.critic_fc3 = nn.Linear(32, 16)

        self.action_head = nn.Linear(40, action_dim)
        self.value_head = nn.Linear(40, 1) # Scalar Value
        self.optimizer = Adas(self.parameters(), lr=0.01)
        
        self.entropies = 0
        self.len = 0
        self.action_probs = []
        self.state_values = []
        self.rewards = []
        self.next_states = []
        self.checkpoint_file = checkpoint_file
    
    def store(self, prob, state_value, reward, next_state):
        self.action_probs.append(prob)
        self.state_values.append(state_value)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.len += 1
    
    def clear(self):
        self.action_probs = []
        self.state_values = []
        self.rewards = []
        self.next_states = []
        self.len = 0
        self.entropies = 0
    
    def get_data(self):
        return self.action_probs, self.state_values, self.rewards, self.next_states
        
    def forward(self, state):
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        # x = F.relu(self.actor_fc3(x))
        action_score = self.action_head(x)
        
        y = F.relu(self.critic_fc1(state))
        y = F.relu(self.actor_fc2(y))
        # y = F.relu(self.actor_fc3(y))
        state_value = self.value_head(y)
        
        probs = F.softmax(action_score, dim = -1)
        return Categorical(probs), state_value

    def optimize(self):
        self.optimizer.step()
        
    def reset_grad(self):
        self.optimizer.zero_grad()

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(self.checkpoint_file, map_location = self.device))