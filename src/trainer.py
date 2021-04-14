import torch
import torch.nn as nn


class Trainer:
    """
    Trainer for an MCTS policy network. Trains the network to minimize
    the difference between the value estimate and the actual returns and
    the difference between the policy estimate and the refined policy estimates
    derived via the tree search.
    """

    def __init__(self, model, learning_rate=0.001):

        self.model = model
        self.lr = learning_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.value_criterion = nn.MSELoss()

        def train(batch):
            obs, search_pis, returns = batch
            obs = torch.from_numpy(obs).to(self.device)
            search_pis = torch.from_numpy(search_pis).to(self.device)
            returns = torch.from_numpy(returns).to(self.device)
            model.reset_grad()
            logits, policy, value = self.model(obs)
            logsoftmax = nn.LogSoftmax(dim=1)
            policy_loss = torch.mean(torch.sum(-search_pis * logsoftmax(logits), dim=1))
            value_loss = self.value_criterion(value, returns.unsqueeze(1))
            loss = policy_loss + value_loss
            loss.backward()
            model.optimize()

            return value_loss.data.numpy(), policy_loss.data.numpy()

        self.train = train
