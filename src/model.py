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
    def __init__(self, num_inputs, state_dim, action_dim, lr = 0.001, 
                 checkpoint_file = 'Models/', is_recurrent=True):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(8, 32 * 3 * 3, 256)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        
        self.actor_fc1 = nn.Linear(32 * 3 * 3, 128)
        self.critic_fc1 = nn.Linear(32 * 3 * 3, 128)
        self.actor_fc1.weight.data = fanin_init(self.actor_fc1.weight.data.size())
        self.critic_fc1.weight.data = fanin_init(self.critic_fc1.weight.data.size())
        
        self.actor_fc2 = nn.Linear(128, 128)
        self.critic_fc2 = nn.Linear(128, 128)
        self.actor_fc2.weight.data = fanin_init(self.actor_fc2.weight.data.size())
        self.critic_fc2.weight.data = fanin_init(self.critic_fc2.weight.data.size())
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    
    def forward(self, inputs):
        # cx = torch.zeros(1, 256).to(self.device)
        # hx = torch.zeros(1, 256).to(self.device)
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        output = x.view(-1, 32 * 3 * 3)
        x = F.relu(self.actor_fc1(output))
        x = F.relu(self.actor_fc2(x))
        # x = F.relu(self.actor_fc3(x))
        action_score = self.action_head(x)
        
        y = F.relu(self.critic_fc1(output))
        y = F.relu(self.actor_fc2(y))
        # y = F.relu(self.actor_fc3(y))
        state_value = self.value_head(y)
        
        probs = F.softmax(action_score, dim = -1)
        return action_score, probs, state_value
    
    def step(self, obs):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        obs = torch.from_numpy(obs).to(self.device)
        _, pi, v = self.forward(obs)

        return pi.detach().numpy(), v.detach().numpy()

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
        
    def optimize(self):
        self.optimizer.step()
        
    def reset_grad(self):
        self.optimizer.zero_grad()

    def save_checkpoint(self, name):
        # print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file + name + '.pt')

    def load_checkpoint(self, name):
        # print('... loading checkpoint ...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(self.checkpoint_file + name + '.pt', map_location = self.device))
        
    
class ActorCritic_2(nn.Module):
    def __init__(self, num_inputs, state_dim, action_dim, lr = 0.001, checkpoint_file = 'Models/', is_recurrent=True):
        super(ActorCritic_2, self).__init__()
        self.recurrent = is_recurrent
        self.action_dim = action_dim
        self.num_inputs = num_inputs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
        return action_score, probs, state_value

    def step(self, obs):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        obs = torch.from_numpy(obs).to(self.device).view(-1, 288)
        _, pi, v = self.forward(obs)

        return pi.detach().numpy(), v.detach().numpy()
    def optimize(self):
        self.optimizer.step()
        
    def reset_grad(self):
        self.optimizer.zero_grad()

    def save_checkpoint(self, name):
        # print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file + name + '.pt')

    def load_checkpoint(self, name):
        # print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file + name + '.pt', map_location = self.device))
        