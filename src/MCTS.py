import logging
import math

import numpy as np
from copy import deepcopy as dcopy
import time

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, agent_pos, player_id, agent_id, depth):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.agent_id = agent_id
        self.agent_pos = agent_pos
        self.player_id = player_id
        self.depth = depth
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        
    def getActionProb(self, canonicalBoard, agent_pos, player_id, agent_id, depth, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = self.game.stringRepresentation(canonicalBoard)
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, agent_pos, player_id, agent_id, depth)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 
                  for a in range(self.game.n_actions)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, agent_pos, player_id, agent_id, depth):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        # self.game.log_state(canonicalBoard)
        # time.sleep(1)
        # print(depth)
        # self.game.log_state(canonicalBoard)
        # time.sleep(2)
        agent_pos = dcopy(agent_pos)
        canonicalBoard = dcopy(canonicalBoard)
        s = self.game.stringRepresentation(canonicalBoard)
        
        self.Es[s] = self.game.getGameEnded(canonicalBoard, player_id, 
                                                agent_id, depth)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            state = dcopy(canonicalBoard)
            agent_pos_ = dcopy(agent_pos)
            if player_id == 1:
                self.game.convert_to_opn_obs(state, agent_pos_)
            state_step = self.game.get_states_for_step(state)
            agent_step = self.game.get_agent_for_step(agent_id, player_id, agent_pos_)
            self.Ps[s], v = self.nnet.step(state_step, agent_step)
            self.Ps[s], v = self.Ps[s][0], v[0]
            # print(self.Ps[s])
            # print(v)
            valids = self.game.getValidMoves(canonicalBoard, agent_pos, 
                                             player_id, agent_id)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
        
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.n_actions):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        # self.game.log_state(canonicalBoard)
        # print(player_id)
        next_s, next_agent_pos, next_player_id, next_agent_id, next_depth = \
            self.game.get_next_state(canonicalBoard, a, agent_pos, player_id,
                                     agent_id, depth)
        # self.game.log_state(next_s)
        v = self.search(next_s, next_agent_pos, next_player_id, next_agent_id, next_depth)
        
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
