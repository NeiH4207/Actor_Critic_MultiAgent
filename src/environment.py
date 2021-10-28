from copy import deepcopy as dcopy
from math import sqrt, acos, pi

from GameBoard.game_board import Screen
from src.utils import flatten
import numpy as np
import time

ScoreBoardID = 0
AgentID = 1
ConquerID = 2
TreasureID = 3
WallID = 4
dx = [1, -1, 0, 0]
dy = [0, 0, -1, 1]

class Player(object):
    
    def __init__(self, ID):
        self.ID = ID
        self.title_score = 0
        self.area_score = 0
        self.treasure_score = 0
        self.old_score = 0
        
    @property
    def total_score(self):
        """
        Returns the total scores consits of title, area and treasure scores
        """
        return self.title_score + self.area_score + self.treasure_score
    
    def reset(self):
        self.title_score = 0
        self.area_score = 0
        self.treasure_score = 0
        self.old_score = 0
        
    def show_scores(self):
        print("Player " + str(self.ID) + ":")
        print("\tTitle Score: {}".format(self.title_score))
        print("\tTreasure Score: {}".format(self.treasure_score))
        print("\tArea Score {}".format(self.area_score))
        print()
        
class Environment(object):

    def __init__(self, input_data = None, show_screen = False, MAX_SIZE = 20):
        self.MAX_SIZE = MAX_SIZE
        self.show_screen = show_screen
        self.data = dcopy(input_data)
        self.n_actions = 8
        self.punish = 0
        self.n_inputs = 7
        self.max_n_agents = 8
        self.max_n_turns = 100
        self.num_players = 2
        self.agent_step_dim = (1 + 2 * self.max_n_agents) * (self.MAX_SIZE ** 2) \
            + 2 * self.max_n_turns + self.num_players
        self.action_dim = self.n_actions
        self.players = [Player(i) for i in range(self.num_players)]
        self.screen = Screen(self)
        self.reset()
        
        '''
        print("Infor map: ")
        print("\tHeight - Width: {}-{}".format(self.height, self.width))
        print("\tNum agents: {}".format(self.n_agents))
        print()
        '''
    
    def reset(self):
        """
        height: height of table
        width: width of table
        score_board: title score in table
        agent_pos: location of agents in table (coord)
        treasure_board: treasures in table
        wall_board: walls in table
        conquer_board: conquered locations of players
        n_turns: number of turns in each game
        n_agents: number of agents

        """
        height, width, score_board, agent_pos, treasure_board, wall_board, \
            conquer_board, n_turns, n_agents = [dcopy(_data) for _data in self.data]
            
        self.score_board = []
        self.agent_board = [[], []]
        self.treasure_board = []
        self.wall_board = []
        self.conquer_board = [[], []]
        self.width = width
        self.height = height
        self.agent_pos = agent_pos
        self.n_turns = n_turns
        self.remaining_turns = n_turns
        self.n_agents = n_agents
        
        for player_ID in range(self.num_players):
            self.players[player_ID].reset()
    
        for _ in range(self.MAX_SIZE):
            self.score_board.append([0] * self.MAX_SIZE)
            self.treasure_board.append([0] * self.MAX_SIZE)
            self.wall_board.append([0] * self.MAX_SIZE)
            for player_ID in range(self.num_players):
                self.agent_board[player_ID].append([0] * self.MAX_SIZE)
                self.conquer_board[player_ID].append([0] * self.MAX_SIZE)

        for i in range(self.height):
            for j in range(self.width):
                self.score_board[i][j] = score_board[i][j]
        
        
        for i in range(self.n_agents):    
            for j in range(self.num_players):
                x, y = self.agent_pos[j][i]
                self.agent_board[j][x][y] = 1
                self.conquer_board[j][x][y] = 1
            
        for x, y in wall_board:
            self.wall_board[x][y] = 1
        
        for i in range(self.MAX_SIZE):
            for j in range(self.MAX_SIZE):
                if i >= self.height or j >= self.width:
                    self.wall_board[i][j] = 1
            
        for x, y, value in treasure_board:
            self.treasure_board[x][y] = value
            
        self.upper_bound_score = np.max(score_board)
        self.lower_bound_score = np.min(score_board)
        self.norm_score_board = dcopy(self.score_board)
        self.norm_treasure_board = dcopy(self.treasure_board)
        self.range_bound = (self.upper_bound_score - self.lower_bound_score)
        self.score_board = (self.score_board - self.lower_bound_score) \
            / self.range_bound
        self.treasure_board /= self.range_bound
        
        self.observation = self.get_observation(0)
            
        title_scores, treasure_scores, area_scores = \
            self.compute_score(self.observation, self.observation)
        
        for player_ID in range(self.num_players):
            self.players[player_ID].title_score = title_scores[player_ID]
            self.players[player_ID].treasure_score = treasure_scores[player_ID]
            self.players[player_ID].area_score = area_scores[player_ID]
            self.players[player_ID].old_score = self.players[player_ID].total_score
        
        self.old_observation = dcopy(self.observation)
        
        if self.show_screen:
            self.screen.setup(self)
    
    def soft_reset(self):
        
        for player_ID in range(self.num_players):
            self.players[player_ID].reset()
            for x in range(self.height):
                for y in range(self.width):       
                    self.agent_board[player_ID][x][y] = 0
                    self.conquer_board[player_ID][x][y] = 0
        
        for i in range(self.n_agents):    
            for j in range(2):
                x, y = self.agent_pos[j][i]
                if self.show_screen:
                    self.screen.reset_square([x, y], -1, 0)
                    
        height, width, _, agent_pos,  _, _, \
            conquer_board, n_turns, n_agents = [dcopy(_data) for _data in self.data]
            
        self.agent_pos = agent_pos
        self.remaining_turns = self.n_turns
        for player_ID in range(self.num_players):
            for agent_ID in range(self.n_agents):
                x, y = self.agent_pos[player_ID][agent_ID]
                self.agent_board[player_ID][x][y] = 1
                self.conquer_board[player_ID][x][y] = 1
    
        self.observation = self.get_observation(0)
        title_scores, treasure_scores, area_scores = \
            self.compute_score(self.observation, self.observation)
        
        for player_ID in range(self.num_players):
            self.players[player_ID].title_score = title_scores[player_ID]
            self.players[player_ID].treasure_score = treasure_scores[player_ID]
            self.players[player_ID].area_score = area_scores[player_ID]
            self.players[player_ID].old_score = self.players[player_ID].total_score
            
        self.old_observation = dcopy(self.observation)
        
        if self.show_screen:
            self.screen.reset()
        
    def stringRepresentation(self, state):
        return hash(str(state))
        
    def getGameEnded(self, board, player, agent, depth):
        if (player == 1  and agent == self.n_agents - 1 and depth == self.n_turns):
            # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
            # player = 1
            title_scores, treasures_scores, area_scores = self.compute_score(board, board)
            diff_score = title_scores[player] + treasures_scores[player] + area_scores[player] +\
                    - title_scores[1 - player] - treasures_scores[1 - player] - area_scores[1 - player]
    
            # for player in range(self.num_players):
            #     self.players[player].title_score = title_scores[player]
            #     self.players[player].treasure_score += treasures_scores[player]
            #     self.players[player].area_score = area_scores[player]
            return diff_score + 1e-7
        return 0
    
    def render(self):
        """
        display game screen
        """
        self.screen.render()
        
    def get_ub_board_size(self):
        """
        Returns upper bound of board size
        """
        return [self.MAX_SIZE, self.MAX_SIZE]
    
    def get_state(self, player):
        state = self.get_observation(player)
        return state
    
    def get_observation(self, player_ID):
        """
        Returns current observation
        """
        
        state = dcopy([
            self.score_board, 
            self.agent_board, 
            self.conquer_board, 
            self.treasure_board, 
            self.wall_board])
        
        if player_ID == 1:
            temp = dcopy(state[1][0])
            state[1][0] = dcopy(state[1][1])
            state[1][1] = temp
            temp = dcopy(state[2][0])
            state[2][0] = dcopy(state[2][1])
            state[2][1] = temp
        return state
    
    def get_symmetric_state(self, state, agent_step):
        
        if np.random.choice([True, False]):
            _r =  np.random.randint(3)
            for i in range(len(state[0])):
                t = state[0][i]
                state[0][i] = np.rot90(t, _r)
            agent_step = np.rot90(agent_step, _r)
        if np.random.choice([True, False]):
            for i in range(len(state[0])):
                t = state[0][i]
                state[0][i] = np.fliplr(t)
            agent_step = np.fliplr(agent_step)
        agent_step = self.get_agents_for_step(agent_step)
        return state[0], agent_step[0]
    
    def convert_to_opn_obs(self, state, agent_pos):
        """
        Returns opponent observation
        """
        temp = state[1][0]
        state[1][0] = state[1][1]
        state[1][1] = temp
        temp = state[2][0]
        state[2][0] = state[2][1]
        state[2][1] = temp
        
        temp = agent_pos[1]
        agent_pos[1] = agent_pos[0]
        agent_pos[0] = temp
        return state, agent_pos
    
    def log_state(self, state):
        # print("Score Board: ")
        # for i in range(self.height):
        #     print(self.norm_score_board[i][:self.width])
        print("Agent Board 1: ", "Agent Board 2: ")
        for i in range(self.height):
            print(['X' if x == 1 else '-' for x in state[AgentID][0][i][:self.width]],
                  ['O' if x == 1 else '-' for x in state[AgentID][1][i][:self.width]])
        # for i in range(self.height):
        #     print()
        print('-----------')
        print("Conquered Agent Board 1: ", "Conquered Agent Board 2: ")
        for i in range(self.height):
            print(['X' if x == 1 else '-' for x in state[ConquerID][0][i][:self.width]],
                  ['O' if x == 1 else '-' for x in state[ConquerID][1][i][:self.width]])
        # for i in range(self.height):
        #     print()
        print('-----------')
    
    def get_states_for_step(self, states):
        states = np.array(flatten(states), dtype = np.float32)\
            .reshape(-1, self.n_inputs, self.MAX_SIZE, self.MAX_SIZE)
        return states

    
    def get_agent_pos(self, player):
        return dcopy(self.agent_pos[player])
    
    def get_agent_pos_all(self):
        return dcopy(self.agent_pos)
    
    def get_valid_moves(self, board, coord):
        # return a fixed size binary vector
        valids = [1] * self.n_actions
        x, y = coord
        for act in range(self.n_actions):
            x1, y1 = self.next_action(x, y, act)
            ''' invalid checking '''
            if x1 < 0 or x1 >= self.height or y1 < 0 or y1 >= self.width:
                valids[act] = 0
            elif self.wall_board[x1][y1] == 1:
                valids[act] = 0
            elif board[AgentID][0][x1][y1] == 1 or board[AgentID][1][x1][y1] == 1:
                valids[act] = 0
        return np.array(valids)
    
    def get_agent_for_step(self, agent_ID, player_ID, agent_coord):
        agent_state = [[], []]
        for ag_id in range(self.max_n_agents):
            for player_ID in range(self.num_players):
                empty_board = []
                for _ in range(self.MAX_SIZE):
                    empty_board.append([0] * self.MAX_SIZE)
                agent_state[player_ID].append(empty_board)
                if ag_id >= self.n_agents: continue
                x, y = agent_coord[player_ID][ag_id]
                agent_state[player_ID][ag_id][x][y] = 1
                    
        index = agent_state[0][agent_ID]
        onehot_nturns = [[0] * self.max_n_turns, [0] * self.max_n_turns]
        onehot_nturns[0][self.n_turns] = 1
        onehot_nturns[1][self.remaining_turns] = 1
        onehot_players = [1, 1]
        if player_ID == 0:
            onehot_players[1] = 0
        else:
            onehot_players[0] = 0
        agent_state = flatten([agent_state, index, onehot_nturns, onehot_players])
        # print(len(agent_state))
        return np.array(agent_state, dtype = np.float32).reshape(-1, self.agent_step_dim)
    
    def get_agents_for_step(self, agents_step):
        agents_step = np.array(agents_step, dtype = np.float32)\
            .reshape(-1, self.agent_step_dim)
        return agents_step
    
    def get_act(act):
        switcher = {
                (1, 0):   0,
                (1, 1):   1,
                (0, 1):   2,
                (-1, 1):  3,
                (-1, 0):  4,
                (-1, -1): 5,
                (0, -1):  6,
                (1, -1):  7,
            }
        return switcher.get(act, 0)
    
    def compute_score_area(self, state, player_ID):
        def is_border(x, y):
            return x <= 0 or x >= self.height - 1 or y <= 0 or y >= self.width - 1
        
        def can_move(x, y):
            return x >= 0 and x < self.height and y >= 0 and y < self.width
        
        def dfs(x, y, visited):
            visited[x][y] = True
            is_closed = True
            if is_border(x, y):
                is_closed = False
            temp_score = abs(score_board[x][y])
            if wall_board[x][y] == 1:
                temp_score = 0
            for i in range(4):
                _x = x + dx[i]
                _y = y + dy[i]
                if can_move(_x, _y) and not visited[_x][_y]:
                   _score = dfs(_x, _y, visited)
                   if _score < 0:
                       is_closed = False
                   else:
                       temp_score += _score
            if not is_closed:
                return -1
            return temp_score
        
        visited = []
        score_board = state[ScoreBoardID]
        conquer_board = state[ConquerID]
        wall_board = state[WallID]
        score = 0
        for i in range(self.height):
            visited.append([False] * self.width)
            for j in range(self.width):
                if conquer_board[player_ID][i][j] == 1:
                    visited[i][j] = True

        for i in range(self.height):
            for j in range(self.width):
                if not visited[i][j]:
                    temp = dfs(i, j, visited)
                    score += max(0, temp)
                    # if score > 0:
                    #     print(score)
                    #     print(visited)
        return score
        
    def compute_score(self, state, old_state):
        """
        
        Parameters
        ----------
        state : object
            state of game.
        old_state : object
            prestate of game.

        Returns
        -------
        title_scores : array
            title scores of players.
        treasure_score : TYPE
            treasure scores of players.
        area_scores : TYPE
            area scores of players.

        """
        treasure_board = state[TreasureID]
        title_scores = [0, 0]
        treasure_score = [0, 0]
        area_scores = [0, 0]
        for i in range(self.height):
            for j in range(self.width):
                if state[ConquerID][0][i][j] == 1:
                    title_scores[0] += state[ScoreBoardID][i][j]
                if state[ConquerID][1][i][j] == 1:
                    title_scores[1] += state[ScoreBoardID][i][j]
                if state[TreasureID][i][j] > 0 and old_state[ConquerID][0][i][j] == 0 \
                        and old_state[ConquerID][1][i][j] == 0:
                    if state[ConquerID][0][i][j] == 1:
                        state[TreasureID][0] += treasure_board[i][j]
                    if state[ConquerID][1][i][j] == 1:
                        state[TreasureID][1] += treasure_board[i][j]
                        
        for player_ID in range(self.num_players):
            area_scores[player_ID] = self.compute_score_area(state, player_ID)
            
        return title_scores, treasure_score, area_scores
    
    def get_score(self, state, old_state, player_ID):
        state = dcopy(state)
        title_scores, treasure_scores, area_scores = self.compute_score(state, old_state)
        result = title_scores[0] + treasure_scores[0] + area_scores[0] \
            - title_scores[1] - treasure_scores[1] - area_scores[1]
        return result
            
    def check_next_action(self, _act, id_agent, agent_pos):
        x, y = agent_pos[id_agent][0], agent_pos[id_agent][1]
        x, y = self.next_action(x, y, _act)
        if not (x >= 0 and x < self.height and y >= 0 and y < self.width):
            return False
        
        return self.wall_board[x][y] == 0
    
    def next_action(self, x, y, act):
        def action(x):
            switcher = {
                0: [1, 0], 1: [1, 1], 2: [0, 1], 3: [-1, 1], 
                4: [-1, 0], 5: [-1, -1], 6: [0, -1], 7: [1, -1]
            }
            return switcher.get(x, [1, 0])
        _action = action(act)
        return [x + _action[0], y + _action[1]]
    
    def angle(self, a1, b1, a2, b2):
        fi = acos((a1 * a2 + b1 * b2) / (sqrt(a1*a1 + b1*b1) * (sqrt(a2*a2 + b2*b2))))
        return fi
    
    def check(self, x0, y0, x, y, act):
        
        def action(x):
            switcher = {
                0: [1, 0], 1: [1, 1], 2: [0, 1], 3: [-1, 1], 
                4: [-1, 0], 5: [-1, -1], 6: [0, -1], 7: [1, -1]
            }
            return switcher.get(x, [1, 0])
        
        a1, b1 = action(act)
        a2, b2 = x - x0, y - y0
        if abs(self.angle(a1, b1, a2, b2)) - 0.0001 <= pi / 3:
            return True
        return False
    
    def predict_spread_scores(self, x, y, state, act):
        score_board, agent_board, conquer_board, treasure_board, wall_board = state
        score = 0
        discount = 0.02
        reduce_negative = 0.02
        p_1 = 1.3
        p_2 = 1
        aux_score = 0
        for i in range(1, min(8, self.remaining_turns)):
            for j in range(max(0, x - i), min(self.height, x + i + 1)):
                new_x = j
                new_y = y - i
                if new_y >= 0:
                    if wall_board[new_x][new_y] == 0: 
                        _sc = treasure_board[new_x][new_y] ** p_1
                        if conquer_board[0][new_x][new_y] != 1:
                            _sc += (max(reduce_negative * score_board[new_x][new_y], score_board[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            score += _sc * discount
                new_x = j
                new_y = y + i
                if new_y  < self.width:
                    if wall_board[new_x][new_y] == 0: 
                        _sc = treasure_board[new_x][new_y] ** p_1
                        if conquer_board[0][new_x][new_y] != 1:
                            _sc += (max(reduce_negative * score_board[new_x][new_y], score_board[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            score += _sc * discount
            for k in range(max(0, y - i), min(self.height, y + i + 1)):
                new_x = x - i
                new_y = k
                if new_x >= 0:
                    if wall_board[new_x][new_y] == 0: 
                        _sc = treasure_board[new_x][new_y] ** p_1
                        if conquer_board[0][new_x][new_y] != 1:
                            _sc += (max(reduce_negative * score_board[new_x][new_y], score_board[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            score += _sc * discount
                new_x = x + i
                new_y = k
                if new_x < self.height:
                    if wall_board[new_x][new_y] == 0: 
                        _sc = treasure_board[new_x][new_y] ** p_1
                        if conquer_board[0][new_x][new_y] != 1:
                            _sc += (max(reduce_negative * score_board[new_x][new_y], score_board[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            score += _sc * discount
            discount *= 0.7
        return score
    
    def soft_step(self, agent_id, state, act, agent_pos, exp = False):
        old_state = dcopy(state)
        old_scores, old_treasures_scores, area_scores = self.compute_score(old_state, old_state)
        old_score = old_scores[0] + area_scores[0] - old_scores[1] - area_scores[1]
        score_board, agent_board, conquer_board, treasure_board, wall_board = state
        x, y = agent_pos[agent_id][0], agent_pos[agent_id][1]     
        _x, _y = self.next_action(x, y, act)
        valid = True
        aux_score = 0
        reward = 0
        if _x >= 0 and _x < self.height and _y >= 0 and _y < self.width and wall_board[_x][_y] == 0:
            if agent_board[0][_x][_y] == 0 and agent_board[1][_x][_y] == 0:
                if conquer_board[1][_x][_y] == 0:
                    agent_board[0][_x][_y] = 1
                    agent_board[0][x][y] = 0
                    conquer_board[0][_x][_y] = 1
                    agent_pos[agent_id][0] = _x
                    agent_pos[agent_id][1] = _y
                else:
                    conquer_board[1][_x][_y] = 0
        else:
            valid = False
            
        title_scores, treasures_scores, area_scores = self.compute_score(state, old_state)
        if valid:
            if exp:
                aux_score = self.predict_spread_scores(_x, _y, state, act)
            else:
                aux_score = 0
            reward = (title_scores[0] + treasures_scores[0] + area_scores [0] + aux_score\
                - title_scores[1] - treasures_scores[1] - area_scores[1] - old_score)
        else:
            reward = - 0.5
            
        return valid, state, reward
    
    def soft_step_2(self, agent_id, state, acts, agent_pos, exp = False):
        ''' storage old state to compute changed scored '''
        old_state = dcopy(state)
        old_scores, old_treasures_scores, area_scores = self.compute_score(old_state, old_state)
        old_score = old_scores[0] + area_scores[0] - old_scores[1] - area_scores[1]
        score_board, agent_board, conquer_board, treasure_board, wall_board = state
        
        ''' get next action '''
        x0, y0 = agent_pos[0][agent_id][0], agent_pos[0][agent_id][1]     
        x1, y1 = self.next_action(x0, y0, acts[0])
        x2, y2 = agent_pos[1][agent_id][0], agent_pos[1][agent_id][1]     
        x3, y3 = self.next_action(x2, y2, acts[1])
        valids = [True, True]
        
        ''' invalid checking '''
        if x1 < 0 or x1 >= self.height or y1 < 0 or y1 >= self.width:
            x1, y1 = x0, y0
            valids[0] = False
        elif wall_board[x1][y1] == 1:
            x1, y1 = x0, y0
            valids[0] = False
        
        if x3 < 0 or x3 >= self.height or y3 < 0 or y3 >= self.width:
            x3, y3 = x2, y2
            valids[1] = False
        elif wall_board[x3][y3] == 1:
            x3, y3 = x2, y2
            valids[1] = False
            
        ''' two actions is invalid'''
        if x1 == x2 and y1 == y2 and x3 == x0 and y3 == y0:
            return state, [0, 0]
        
        ''' conflict to unique square '''
        if x1 == y1 and x3 == y3:
            return state, [0, 0]
        
        ''' go to conquered square '''
        if agent_board[0][x1][y1] == 1 or agent_board[1][x1][y1] == 1:
            x1, y1 = x0, y0
            valids[0] = False
            
        if agent_board[0][x3][y3] == 1 or agent_board[1][x3][y3] == 1:
            x3, y3 = x2, y2
            valids[1] = False
        
        ''' fit actions '''
        if conquer_board[1][x1][y1] == 0:
            agent_board[0][x1][y1] = 1
            agent_board[0][x0][y0] = 0
            conquer_board[0][x1][y1] = 1
            agent_pos[0][agent_id][0] = x1
            agent_pos[0][agent_id][1] = y1
        else:
            conquer_board[1][x1][y1] = 0
            
        if conquer_board[0][x3][y3] == 0:
            agent_board[1][x3][y3] = 1
            agent_board[1][x2][y2] = 0
            conquer_board[1][x3][y3] = 1
            agent_pos[1][agent_id][0] = x3
            agent_pos[1][agent_id][1] = y3
        else:
            conquer_board[1][x1][y1] = 0
            
        title_scores, treasures_scores, area_scores = self.compute_score(state, old_state)
        
        reward = (title_scores[0] + treasures_scores[0] + area_scores[0]\
            - title_scores[1] - treasures_scores[1] - area_scores[1] - old_score)
        
        rewards = [reward, - reward]
        if not valids[0]:
            rewards[0] -= 1
        if not valids[1]:
            rewards[1] -= 1
        
        return state, rewards
    
    def soft_step_(self, agent_id, state, act, agent_pos):
        score_board, agent_board, conquer_board, treasure_board, wall_board = state
        x, y = agent_pos[agent_id][0], agent_pos[agent_id][1]     
        _x, _y = self.next_action(x, y, act)
        if _x >= 0 and _x < self.height and _y >= 0 and _y < self.width and wall_board[_x][_y] == 0:
            if agent_board[0][_x][_y] == 0 and agent_board[1][_x][_y] == 0:
                if conquer_board[1][_x][_y] == 0:
                    agent_board[0][_x][_y] = 1
                    agent_board[0][x][y] = 0
                    conquer_board[0][_x][_y] = 1
                    agent_pos[agent_id][0] = _x
                    agent_pos[agent_id][1] = _y
                else:
                    conquer_board[1][_x][_y] = 0
                    
        return state
    
    def get_next_action_pos(self, action_1, action_2):
        new_pos = [[], []]
        is_valid_action = [[True] * self.n_agents, [True] * self.n_agents]
        
        for i in range(self.n_agents):
            x, y = self.agent_pos[0][i][0], self.agent_pos[0][i][1]
            new_pos[0].append(self.next_action(x, y, action_1[i]))
            x, y = self.agent_pos[1][i][0], self.agent_pos[1][i][1]
            new_pos[1].append(self.next_action(x, y, action_2[i]))
        
        for i in range(self.n_agents):
            x, y = new_pos[0][i]
            if (x < 0 or x >= self.height or y < 0 or y >= self.width):
                is_valid_action[0][i] = False
                new_pos[0][i] = dcopy(self.agent_pos[0][i])
            elif self.wall_board[x][y] == 1:
                is_valid_action[0][i] = False
                new_pos[0][i] = dcopy(self.agent_pos[0][i])
            
        for i in range(self.n_agents):
            x, y = new_pos[1][i]
            if (x < 0 or x >= self.height or y < 0 or y >= self.width):
                is_valid_action[1][i] = False
                new_pos[1][i] = dcopy(self.agent_pos[1][i])
            elif self.wall_board[x][y] == 1:
                is_valid_action[1][i] = False
                new_pos[1][i] = dcopy(self.agent_pos[1][i])
            
        """ create connect matrix """
        connected_matrix = []
        for j in range(2 * self.n_agents):
            connected_matrix.append([0] * (2 * self.n_agents))
            
        for i in range(2 * self.n_agents):
            X = new_pos[0][i] if i < self.n_agents else new_pos[1][i - self.n_agents]
            for j in range(2 * self.n_agents):
                if i == j: continue
                Y = self.agent_pos[0][j] if j < self.n_agents \
                    else self.agent_pos[1][j - self.n_agents]
                if X[0] == Y[0] and X[1] == Y[1]:
                    connected_matrix[i][j] = 1
                        
        """ handle conflict action, 1 square together"""
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if new_pos[0][i][0] == new_pos[1][j][0] and\
                    new_pos[0][i][1] == new_pos[1][j][1]:
                    is_valid_action[0][i] = False
                    is_valid_action[1][j] = False
                    new_pos[0][i] = dcopy(self.agent_pos[0][i])
                    new_pos[1][j] = dcopy(self.agent_pos[1][j])
                if i < j and new_pos[0][i][0] == new_pos[0][j][0] and\
                    new_pos[0][i][1] == new_pos[0][j][1]:
                    is_valid_action[0][i] = is_valid_action[0][j] = False
                    new_pos[0][i] = dcopy(self.agent_pos[0][i])
                    new_pos[0][j] = dcopy(self.agent_pos[0][j])
                if i < j and new_pos[1][i][0] == new_pos[1][j][0] and\
                    new_pos[1][i][1] == new_pos[1][j][1]:
                    is_valid_action[1][i] = is_valid_action[1][j] = False
                    new_pos[1][i] = dcopy(self.agent_pos[1][i])
                    new_pos[1][j] = dcopy(self.agent_pos[1][j])
        
        """ handle the clique """
        for i in range(2 * self.n_agents):
            if i < self.n_agents:
                if not is_valid_action[0][i]:
                    continue
            elif not is_valid_action[1][i - self.n_agents]:
                continue
            u = i
            stk = [u]
            visited = [False] * (2 * self.n_agents)
            visited[u] = True
            
            for _ in range(2 * self.n_agents):
                for j in range(2 * self.n_agents):
                    if connected_matrix[u][j] == 1:
                        stk.append(j)
                        is_clique = False
                        if j < self.n_agents:
                            if not is_valid_action[0][j]: is_clique = True
                        elif not is_valid_action[1][j - self.n_agents]:
                            is_clique = True
                            
                        if visited[j]:
                            is_clique = True
                            
                        if is_clique:
                            for id in stk:
                                if id < self.n_agents:
                                    is_valid_action[0][id] = False
                                    new_pos[0][id] = dcopy(self.agent_pos[0][id])
                                else:
                                    is_valid_action[1][id - self.n_agents] = False
                                    new_pos[1][id - self.n_agents] = \
                                        dcopy(self.agent_pos[1][id - self.n_agents])
                            stk = []
                            break
                        u = j
                        visited[j] = True
        
        """ handle the conflict remove action """
        for i in range(2 * self.n_agents):
            u = i
            stk = []
            visited = [False] * (2 * self.n_agents)
            visited[u] = True
            if i < self.n_agents:
                if not is_valid_action[0][i]:
                    continue
            elif not is_valid_action[1][i - self.n_agents]:
                continue
            
            for _ in range(2 * self.n_agents):
                for j in range(2 * self.n_agents):
                    if connected_matrix[u][j] == 1:
                        congested = False
                        if j < self.n_agents:
                            x, y = new_pos[0][j]
                            if self.conquer_board[1][x][y] == 1 or\
                                not is_valid_action[0][j]:
                                congested = True
                        else:
                            x, y = new_pos[1][j - self.n_agents]
                            if self.conquer_board[0][x][y] == 1 or\
                                not is_valid_action[1][j - self.n_agents]:
                                congested = True
                                
                        if visited[j]:
                            congested = True
                            
                        visited[j] = True
                        
                        if congested:
                            for id in stk:
                                if id < self.n_agents:
                                    is_valid_action[0][id] = False
                                    new_pos[0][id] = dcopy(self.agent_pos[0][id])
                                else:
                                    is_valid_action[1][id - self.n_agents] = False
                                    new_pos[1][id - self.n_agents] = \
                                        dcopy(self.agent_pos[1][id - self.n_agents])
                            stk = []
                            break
                        stk.append(j)
                        u = j
                if len(stk) == 0:
                    break
        
        return new_pos, is_valid_action
    
    def step(self, action_1, action_2, render = False):
        new_pos, is_valid_action = self.get_next_action_pos(action_1, action_2)
        
        # render before action
        for i in range(self.n_agents):
            if is_valid_action[0][i]:
                x, y = new_pos[0][i]
                if self.conquer_board[1][x][y] == 0:
                    if self.agent_pos[0][i][0] != new_pos[0][i][0] \
                        or self.agent_pos[0][i][1] != new_pos[0][i][1]:
                        self.agent_board[0][self.agent_pos[0][i][0]][self.agent_pos[0][i][1]] = 0
                        self.agent_board[0][x][y] = 0
                        if render:
                            self.screen.redraw_squares(
                                self.agent_pos[0][i][0], self.agent_pos[0][i][1], 0)
                                      
            if is_valid_action[1][i]:
                x, y = new_pos[1][i]
                if self.conquer_board[0][x][y] == 0 :
                    if self.agent_pos[1][i][0] != new_pos[1][i][0] \
                        or self.agent_pos[1][i][1] != new_pos[1][i][1]:
                        self.agent_board[1][x][y] = 0
                        self.agent_board[1][self.agent_pos[1][i][0]][self.agent_pos[1][i][1]] = 0
                        if render:
                            self.screen.redraw_squares(
                                self.agent_pos[1][i][0], self.agent_pos[1][i][1], 1)
                        
        # render after action
        for i in range(self.n_agents):
            for j in range(2):
                if is_valid_action[j][i]:
                    x, y = new_pos[j][i]
                    if self.conquer_board[1 - j][x][y] == 1:
                        self.conquer_board[1 - j][x][y] = 0
                        if render:
                            self.screen.reset_square([x, y], -1)
                        new_pos[j][i] = dcopy(self.agent_pos[j][i])
                    else:
                        self.conquer_board[j][x][y] = 1
                        self.agent_board[j][x][y] = 1
      
            self.compute_score(self.observation, self.old_observation)
            
        if render: self.render()  
        for i in range(self.n_agents):
            self.agent_pos[0][i] = [new_pos[0][i][0], new_pos[0][i][1]]
            self.agent_pos[1][i] = [new_pos[1][i][0], new_pos[1][i][1]]
            
        self.observation = self.get_observation(0)
        
        title_scores, treasure_scores, area_scores = \
            self.compute_score(self.observation, self.old_observation)
            
        if render: self.render()
        for player_ID in range(self.num_players):
            self.players[player_ID].title_score = title_scores[player_ID]
            self.players[player_ID].treasure_score += treasure_scores[player_ID]
            self.players[player_ID].area_score = area_scores[player_ID]
            
        self.old_observation = dcopy(self.observation)
        if render:
            for player_id in range(self.num_players):
                for agent_ID in range(self.n_agents):
                    coord = self.agent_pos[player_id][agent_ID]
                    self.screen.reset_square(coord, player_id, agent_ID)
            self.screen.show_score()
        
        
        reward = (self.players[0].total_score - self.players[1].total_score - \
            self.players[0].old_score + self.players[1].old_score)
        for player_ID in range(self.num_players):
            self.players[player_ID].old_score = self.players[player_ID].total_score
            # self.players[player_ID].show_scores()
            
        _board = dcopy(self.observation)
        _board[0] = self.norm_score_board
        title_scores, treasure_scores, area_scores = \
            self.compute_score(_board, _board)

        for player in range(self.num_players):
            self.players[player].title_score = title_scores[player]
            self.players[player].treasure_score += treasure_scores[player]
            self.players[player].area_score = area_scores[player]
            
        if render: 
            self.screen.show_score()
            self.render()
        
        self.remaining_turns -= 1
        terminate = (self.remaining_turns == 0)
        # if terminate:
        #     reward = 1 if self.players[0].total_score > self.players[1].total_score else -1
        # else:
        #     reward = 0.8 if reward > 0 else -0.8
            
        return [self.observation, reward, terminate, self.remaining_turns]

    def get_next_state(self, state, action, agent_pos, player_ID, 
                       agent_ID, depth, render = False):
        # self.log_state(state)
        # print(agent_pos, player_ID, agent_ID, depth)
        score_board, agent_board, conquer_board, treasure_board, wall_board = state
        x, y = agent_pos[player_ID][agent_ID]
        _x, _y = (self.next_action(x, y, action))
        
        if _x >= 0 and _x < self.height and _y >= 0 and _y < self.width and wall_board[_x][_y] == 0:
            if agent_board[0][_x][_y] == 0 and agent_board[1][_x][_y] == 0:
                if conquer_board[1 - player_ID][_x][_y] == 0:
                    if render: 
                        self.screen.redraw_squares(x, y, player_ID)
                        self.screen.reset_square([_x, _y], player_ID)
                        
                    agent_board[player_ID][_x][_y] = 1
                    agent_board[player_ID][x][y] = 0
                    conquer_board[player_ID][_x][_y] = 1
                    treasure_board[_x][_y] = 0
                    agent_pos[player_ID][agent_ID] = [_x, _y]
                else:
                    conquer_board[1][_x][_y] = 0
                    if render:
                        self.screen.reset_square([_x, _y], -1)
        player_ID = 1 - player_ID
        if player_ID == 0:
            agent_ID += 1
            if agent_ID == self.n_agents:
                agent_ID = 0
                depth += 1
                _board = dcopy(state)
                _board[0] = self.norm_score_board
                title_scores, treasure_scores, area_scores = \
                    self.compute_score(_board, _board)
        
                for player in range(self.num_players):
                    self.players[player].title_score = title_scores[player]
                    self.players[player].treasure_score += treasure_scores[player]
                    self.players[player].area_score = area_scores[player]
                if render:
                    self.remaining_turns -= 1
                    self.screen.show_score()
                    self.render()
                    
        state = [score_board, agent_board, conquer_board, treasure_board, wall_board]
        # self.convert_to_opn_obs(state, agent_pos)
        # self.log_state(state)
        # if sum(flatten(agent_board)) > 2:
        #     print(agent_pos, player_ID, agent_ID, depth)
        #     print([x, y], [_x, _y])
        #     print('--------------------Warning! ', sum(flatten(agent_board)))
        return state, agent_pos, player_ID, agent_ID, depth
            
    def get_return(self, state, old_state, player_ID):
        return 1 if self.get_score(state, old_state, player_ID) >= 0 else -1
    
    def is_done_state(self, state, depth):
        return depth >= 2 * (1 + self.n_turns) * self.n_agents
    