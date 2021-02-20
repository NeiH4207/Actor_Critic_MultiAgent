from copy import deepcopy as dcopy
from math import sqrt, acos, pi
from GameBoard.game_board import Screen
from src.utils import flatten
import numpy as np

class Environment(object):

    def __init__(self, input_data = None, show_screen = False, MAX_SIZE = 20):
        self.MAX_SIZE = MAX_SIZE
        self.walls_matrix = []
        self.n_actions = 9
        self.player_1 = 0
        self.player_2 = 1
        self.score_mine = 0
        self.score_opponent = 0
        self.old_score = 0
        self.punish = 0
        self.n_inputs = 8
        self.data = dcopy(input_data)
        self.show_screen = show_screen
        self.screen = Screen(show_screen)
        self.reset()
    
    def render(self):
        self.screen.render()
    
    def reset(self):
        self.score_matrix = []
        self.normalized_score_matrix = []
        self.agents_matrix = [[], []]
        self.treasures_matrix = []
        self.walls_matrix = []
        self.conquer_matrix = [[], []]
        self.treasure_score = [0, 0]
        self.score_mine = 0
        self.score_opponent = 0
        self.old_score = 0
        self.preprocess()
        
    def soft_reset(self):
        
        self.score_mine = 0
        self.score_opponent = 0
        self.old_score = 0
        for i in range(self.n_agents):    
            for j in range(2):
                x, y = self.agent_pos[j][i]
                if self.show_screen:
                    self.screen.reset_square(x, y, 0)
        self.agents_matrix = [[], []]
        self.conquer_matrix = [[], []]
        self.treasure_score = [0, 0]
        height, width, score_matrix, agent_pos,  treasures, walls, \
            conquer_matrix, n_turns, n_agents = [dcopy(_data) for _data in self.data]
        self.agent_pos = agent_pos
        self.remaining_turns = n_turns
        self.n_agents = n_agents
        self.n_turns = n_turns
    
        for i in range(self.MAX_SIZE):
            self.agents_matrix[0].append([0] * self.MAX_SIZE)
            self.agents_matrix[1].append([0] * self.MAX_SIZE)
            self.conquer_matrix[0].append([0] * self.MAX_SIZE)
            self.conquer_matrix[1].append([0] * self.MAX_SIZE)
            
        for i in range(self.n_agents):    
            for j in range(2):
                x, y = self.agent_pos[j][i]
                self.agents_matrix[j][x][y] = 1
                self.conquer_matrix[j][x][y] = 1
        self.observation = [self.score_matrix, self.agents_matrix, \
                           self.conquer_matrix, self.treasures_matrix, self.walls_matrix]
        
        for x, y, value in treasures:
            self.treasures_matrix[x][y] = value
            
        if self.show_screen:
            self.screen.create_board(self.height, self.width, self.observation)
        
    def preprocess(self):
        height, width, score_matrix, agent_pos,  treasures, walls, \
            conquer_matrix, n_turns, n_agents = [dcopy(_data) for _data in self.data]
        
        self.width = width
        self.height = height
        self.agent_pos = agent_pos
        self.remaining_turns = n_turns
        self.n_agents = n_agents
        self.n_turns = n_turns
    
        for i in range(self.MAX_SIZE):
            self.score_matrix.append([0] * self.MAX_SIZE)
            self.agents_matrix[0].append([0] * self.MAX_SIZE)
            self.agents_matrix[1].append([0] * self.MAX_SIZE)
            self.conquer_matrix[0].append([0] * self.MAX_SIZE)
            self.conquer_matrix[1].append([0] * self.MAX_SIZE)
            self.treasures_matrix.append([0] * self.MAX_SIZE)
            self.walls_matrix.append([0] * self.MAX_SIZE)
            
        for i in range(self.height):
            for j in range(self.width):
                self.score_matrix[i][j] = score_matrix[i][j]
                
        maximum = np.max(score_matrix)
        minimum = np.min(score_matrix)
        self.normalized_score_matrix = (self.score_matrix[i][j] - minimum) / (maximum - minimum)
            
        for i in range(self.n_agents):    
            for j in range(2):
                x, y = self.agent_pos[j][i]
                self.agents_matrix[j][x][y] = 1
                self.conquer_matrix[j][x][y] = 1
            
        for x, y in walls:
            self.walls_matrix[x][y] = 1
        
        for i in range(self.MAX_SIZE):
            for j in range(self.MAX_SIZE):
                if(i >= self.height or j >= self.width):
                    self.walls_matrix[i][j] = 1
            
        for x, y, value in treasures:
            self.treasures_matrix[x][y] = value
            
        self.observation = [self.score_matrix, self.agents_matrix, \
                                    self.conquer_matrix, self.treasures_matrix, self.walls_matrix]
            
        if self.show_screen:
            self.screen.create_board(self.height, self.width, self.observation)
            
        self.observation_dim = len(self.get_state(0, 0))
        self.action_dim = 9
    
    def get_state(self, player, agent_id):
        state = dcopy(self.get_observation(player))
        state.append(self.get_agent_state(agent_id, self.agent_pos[player]))
        return state
    
    def get_observation(self, player):
        state = dcopy([self.score_matrix, self.agents_matrix, self.conquer_matrix, 
                       self.treasures_matrix, self.walls_matrix])
        if player == 1:
            temp = dcopy(state[1][0])
            state[1][0] = dcopy(state[1][1])
            state[1][1] = temp
            temp = dcopy(state[2][0])
            state[2][0] = dcopy(state[2][1])
            state[2][1] = temp
        return state
    
    def get_obs_for_states(self, states):
        states = np.array(flatten(states), dtype = np.float32)\
            .reshape(-1, self.n_inputs, self.MAX_SIZE, self.MAX_SIZE)
        for state in states:
            state[0] = self.normalized_score_matrix
        return states
    
    
    def get_agent_pos(self, player):
        return dcopy(self.agent_pos[player])
    
    def get_agent_state(self, agent_id, agent_pos):
        agent_state = []
        for i in range(self.MAX_SIZE):
            agent_state.append([0] * self.MAX_SIZE)
        x, y = agent_pos[agent_id]
        agent_state[x][y] = 1
        return agent_state
    
    def get_act(act):
        switcher = {
                (0, 0):   0,
                (1, 0):   1,
                (1, 1):   2,
                (0, 1):   3,
                (-1, 1):  4,
                (-1, 0):  5,
                (-1, -1): 6,
                (0, -1):  7,
                (1, -1):  8,
            }
        return switcher.get(act, 0)
    
    def compute_score_area(self, state, player):
        area_matrix = []
        score_matrix, agent_matrix, conquer_matrix, treasures_matrix, walls_matrix = state
        visit = []
        score = 0
        for i in range(self.MAX_SIZE):
            visit.append([0] * self.MAX_SIZE)
            area_matrix.append([0] * self.MAX_SIZE)
            for j in range(self.MAX_SIZE):
                visit[i][j] = conquer_matrix[player][i][j]
            
        def is_border(x, y):
            return x <= 0 or x >= self.height - 1 or y <= 0 or y >= self.width - 1
        
        def can_move(x, y):
            return x >= 0 and x < self.height and y >= 0 and y < self.width \
                and conquer_matrix[player][x][y] != 1
        
        def dfs(x, y):
            visit[x][y] = 1
            area_matrix[x][y] = 1
            temp_score = abs(score_matrix[x][y])
            if(walls_matrix[x][y] == 1):
                area_matrix[x][y] = 0
                temp_score = 0
            if is_border(x, y):
                area_matrix[x][y] = 0
                return -1
            dx = [1, -1, 0, 0]
            dy = [0, 0, -1, 1]
            ok = True
            for i in range(4):
                if can_move(x + dx[i], y + dy[i]) and visit[x + dx[i]][y + dy[i]] == 0:
                   _score = dfs(x + dx[i], y + dy[i])
                   if _score == -1:
                       ok = False
                   else:
                       temp_score += _score
            if ok == False:
                area_matrix[x][y] = 0
                return -1
            return temp_score
        
        
        for i in range(self.MAX_SIZE):
            for j in range(self.MAX_SIZE):
                if visit[i][j] == 0:
                    temp = dfs(i, j)
                    score += max(0, temp)
                    
        return score, area_matrix
        
    def compute_score(self, state):
        score_matrix, agent_matrix, conquer_matrix, treasures_matrix, walls_matrix = state
        score_title = [0, 0]
        treasure_score = [0, 0]
        for i in range(self.MAX_SIZE):
            for j in range(self.MAX_SIZE):
                if(conquer_matrix[0][i][j] == 1):
                    score_title[0] += score_matrix[i][j]
                if(conquer_matrix[1][i][j] == 1):
                    score_title[1] += score_matrix[i][j]
                if(treasures_matrix[i][j] > 0):
                    if(conquer_matrix[0][i][j] == 1):
                        treasure_score[0] += treasures_matrix[i][j]
                        treasures_matrix[i][j] = 0
                    if(conquer_matrix[1][i][j] == 1):
                        treasure_score[1] += treasures_matrix[i][j]
                        treasures_matrix[i][j] = 0
        score_area_A, area_matrix_1 = self.compute_score_area(state, 0)
        score_area_B, area_matrix_2 = self.compute_score_area(state, 1)
            
        score_A = score_title[0] + score_area_A
        score_B = score_title[1] + score_area_B
        return [score_A, score_B], treasure_score, area_matrix_1
    
    def get_score(self, state, player_ID):
        state = dcopy(state)
        state.pop()
        scores, treasure_scores, _ = self.compute_score(state)
        result = scores[0] + treasure_scores[0] - scores[1] - treasure_scores[1]
        return 1 if result > 0 else -1
            
    def check_next_action(self, _act, id_agent, agent_pos):
        x, y = agent_pos[id_agent][0], agent_pos[id_agent][1]
        x, y = self.next_action(x, y, _act)
        if not (x >= 0 and x < self.height and y >= 0 and y < self.width):
            return False
        
        return self.walls_matrix[x][y] == 0
    
    def next_action(self, x, y, act):
        def action(x):
            switcher = {
                0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1],
                4: [-1, 1], 5: [-1, 0], 6: [-1, -1], 7: [0, -1], 8: [1, -1]
            }
            return switcher.get(x, [0, 0])
        _action = action(act)
        return [x + _action[0], y + _action[1]]
    
    def angle(self, a1, b1, a2, b2):
        fi = acos((a1 * a2 + b1 * b2) / (sqrt(a1*a1 + b1*b1) * (sqrt(a2*a2 + b2*b2))))
        return fi
    
    def check(self, x0, y0, x, y, act):
        
        def action(x):
            switcher = {
                0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1],
                4: [-1, 1], 5: [-1, 0], 6: [-1, -1], 7: [0, -1], 8: [1, -1]
            }
            return switcher.get(x, [0, 0])
        
        a1, b1 = action(act)
        a2, b2 = x - x0, y - y0
        if abs(self.angle(a1, b1, a2, b2)) - 0.0001 <= pi / 3:
            return True
        return False
    
    def predict_spread_scores(self, x, y, state, predict, act, area_matrix):
        score_matrix, agents_matrix, conquer_matrix, treasures_matrix, walls_matrix = state
        score = 0
        discount = 0.02
        reduce_negative = 0.02
        p_1 = 1.3
        p_2 = 1
        for i in range(1, min(8, self.remaining_turns)):
            for j in range(max(0, x - i), min(self.height, x + i + 1)):
                new_x = j
                new_y = y - i
                if new_y >= 0:
                    if walls_matrix[new_x][new_y] == 0: 
                        _sc = treasures_matrix[new_x][new_y] ** p_1
                        if(conquer_matrix[0][new_x][new_y] != 1):
                            _sc += (max(reduce_negative * score_matrix[new_x][new_y], score_matrix[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            if area_matrix[new_x][new_y] == 0:
                                score += _sc * discount
                new_x = j
                new_y = y + i
                if new_y  < self.width:
                    if walls_matrix[new_x][new_y] == 0: 
                        _sc = treasures_matrix[new_x][new_y] ** p_1
                        if(conquer_matrix[0][new_x][new_y] != 1):
                            _sc += (max(reduce_negative * score_matrix[new_x][new_y], score_matrix[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            if area_matrix[new_x][new_y] == 0:
                                score += _sc * discount
            for k in range(max(0, y - i), min(self.height, y + i + 1)):
                new_x = x - i
                new_y = k
                if new_x >= 0:
                    if walls_matrix[new_x][new_y] == 0: 
                        _sc = treasures_matrix[new_x][new_y] ** p_1
                        if(conquer_matrix[0][new_x][new_y] != 1):
                            _sc += (max(reduce_negative * score_matrix[new_x][new_y], score_matrix[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            if area_matrix[new_x][new_y] == 0:
                                score += _sc * discount
                new_x = x + i
                new_y = k
                if new_x < self.height:
                    if walls_matrix[new_x][new_y] == 0: 
                        _sc = treasures_matrix[new_x][new_y] ** p_1
                        if(conquer_matrix[0][new_x][new_y] != 1):
                            _sc += (max(reduce_negative * score_matrix[new_x][new_y], score_matrix[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            if area_matrix[new_x][new_y] == 0:
                                score += _sc * discount
            discount *= 0.7
        return score
    
    def fit_action(self, agent_id, state, act, agent_pos, predict = True):
        score_matrix, agents_matrix, conquer_matrix, treasures_matrix, walls_matrix = dcopy(state)
        x, y = agent_pos[0][agent_id][0], agent_pos[0][agent_id][1]     
        new_pos = (self.next_action(x, y, act))
        _x, _y = new_pos
        aux_score = 0
        valid = True
        punish = 0
        if _x >= 0 and _x < self.height and _y >= 0 and _y < self.width and walls_matrix[_x][_y] == 0:
            if agents_matrix[0][_x][_y] == 0 and agents_matrix[1][_x][_y] == 0:
                if conquer_matrix[1][_x][_y] == 0:
                    agents_matrix[0][_x][_y] = 1
                    agents_matrix[0][x][y] = 0
                    conquer_matrix[0][_x][_y] = 1
                    agent_pos[0][agent_id][0] = _x
                    agent_pos[0][agent_id][1] = _y
                    aux_score += 1
                else:
                    conquer_matrix[1][_x][_y] = 0
                    aux_score -= 0.5
                    punish += self.MAX_SIZE
        else:
            valid = False
            
        state = [score_matrix, agents_matrix, conquer_matrix, treasures_matrix, walls_matrix]
        score_1, score_2, treasures_score_1, treasures_score_2, area_matrix = self.compute_score(state)
            
        if(predict is False):
            aux_score = 0
        else:
            if valid:
                aux_score += self.predict_scores(_x, _y, state, predict, act, area_matrix)
            
        return valid, state, agent_pos[0], score_1 + treasures_score_1 - score_2 - treasures_score_2 + aux_score
    
    def soft_step(self, agent_id, state, act, agent_pos, predict = True):
        score_matrix, agents_matrix, conquer_matrix, treasures_matrix, walls_matrix = dcopy(state)
        x, y = agent_pos[agent_id][0], agent_pos[agent_id][1]     
        new_pos = (self.next_action(x, y, act))
        _x, _y = new_pos
        aux_score = 0
        valid = True
        punish = 0
        if _x >= 0 and _x < self.height and _y >= 0 and _y < self.width and walls_matrix[_x][_y] == 0:
            if agents_matrix[0][_x][_y] == 0 and agents_matrix[1][_x][_y] == 0:
                if conquer_matrix[1][_x][_y] == 0:
                    agents_matrix[0][_x][_y] = 1
                    agents_matrix[0][x][y] = 0
                    conquer_matrix[0][_x][_y] = 1
                    agent_pos[agent_id][0] = _x
                    agent_pos[agent_id][1] = _y
                    aux_score += 1
                else:
                    conquer_matrix[1][_x][_y] = 0
                    aux_score -= 0.5
                    punish += self.MAX_SIZE
        else:
            valid = False
            
        state = [score_matrix, agents_matrix, conquer_matrix, treasures_matrix, walls_matrix]
        scores, treasures_scores, area_matrix = self.compute_score(state)
            
        if(predict is False):
            aux_score = 0
        else:
            if valid:
                aux_score += self.predict_spread_scores(_x, _y, state, predict, act, area_matrix)
        reward = scores[0] + treasures_scores[0] - scores[1] - treasures_scores[1] + aux_score
        
        return valid, state, reward
    
    def soft_step_(self, state, action):
        score_matrix, agents_matrix, conquer_matrix, treasures_matrix, walls_matrix,\
            agent = dcopy(state)
        x, y = -1, -1
        for i in range(self.height):
            for j in range(self.width):
                if agent[i][j] == 1:
                    x, y = i, j
        new_pos = (self.next_action(x, y, action))
        _x, _y = new_pos
        valid = True
        if _x >= 0 and _x < self.height and _y >= 0 and _y < self.width and walls_matrix[_x][_y] == 0:
            if agents_matrix[0][_x][_y] == 0 and agents_matrix[1][_x][_y] == 0:
                if conquer_matrix[1][_x][_y] == 0:
                    agents_matrix[0][_x][_y] = 1
                    agents_matrix[0][x][y] = 0
                    conquer_matrix[0][_x][_y] = 1
                    treasures_matrix[_x][_y] = 0
                    agent[x][y] = 0
                    agent[_x][_y] = 1
                else:
                    conquer_matrix[1][_x][_y] = 0
        else:
            valid = False
        
        
        state = [score_matrix, agents_matrix, conquer_matrix, treasures_matrix, walls_matrix, agent]         
        return state
    
    def get_next_action_pos(self, action_1, action_2):
        point_punish = 30
        punish = 0
        new_pos = [[], []]
        checked = [[False] * self.n_agents, [False] * self.n_agents]
        
        for i in range(self.n_agents):
            x, y = self.agent_pos[0][i][0], self.agent_pos[0][i][1]
            new_pos[0].append(self.next_action(x, y, action_1[i]))
            x, y = self.agent_pos[1][i][0], self.agent_pos[1][i][1]
            new_pos[1].append(self.next_action(x, y, action_2[i]))
        
        for i in range(self.n_agents):
            x, y = new_pos[0][i]
            if not (x >= 0 and x < self.height and y >= 0 and y < self.width):
                checked[0][i] = True
                new_pos[0][i] = dcopy(self.agent_pos[0][i])
                punish += point_punish
            elif self.walls_matrix[x][y] == 1:
                checked[0][i] = True
                new_pos[0][i] = dcopy(self.agent_pos[0][i])
                punish += point_punish
            
        for i in range(self.n_agents):
            x, y = new_pos[1][i]
            if not (x >= 0 and x < self.height and y >= 0 and y < self.width):
                checked[1][i] = True
                new_pos[1][i] = dcopy(self.agent_pos[1][i])
            elif self.walls_matrix[x][y] == 1:
                checked[1][i] = True
                new_pos[1][i] = dcopy(self.agent_pos[1][i])
            
        # create connect matrix
        connected_matrix = []
        for j in range(2 * self.n_agents):
            connected_matrix.append([0] * (2 * self.n_agents))
            
        for i in range(2 * self.n_agents):
            X = new_pos[0][i] if i < self.n_agents else new_pos[1][i % self.n_agents]
            for j in range(2 * self.n_agents):
                if i == j:
                    continue
                Y = self.agent_pos[0][j] if j < self.n_agents else self.agent_pos[1][j % self.n_agents]
                if X[0] == Y[0] and X[1] == Y[1]:
                    connected_matrix[i][j] = 1
                        
        # if conflict action to 1 square
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                for k in range(2):
                    for l in range(k, 2):
                        if i == j and k == l: continue
                        if new_pos[k][i][0] == new_pos[l][j][0] and\
                            new_pos[k][i][1] == new_pos[l][j][1]:
                            checked[k][i] = True
                            checked[l][j] = True
                            if i != j:
                                punish += point_punish
        
        
        for i in range(self.n_agents):
            for j in range(2):
                if checked[j][i]:
                    new_pos[0][i] = dcopy(self.agent_pos[0][i])
                    
        # find the clique
        for i in range(2 * self.n_agents):
            if i < self.n_agents:
                if checked[0][i]:
                    continue
            elif checked[1][i - self.n_agents]:
                continue
            u = i
            stk = [u]
            visited = [False] * (2 * self.n_agents)
            visited[u] = True
            
            for _ in range(2 * self.n_agents):
                for j in range(2 * self.n_agents):
                    if connected_matrix[u][j] == 1 and u != j:
                        stk.append(j)
                        is_clique = False
                        if j < self.n_agents:
                            if checked[0][j]: is_clique = True
                        elif checked[1][j - self.n_agents]:
                            is_clique = True
                        if visited[j]:
                            is_clique = True
                            
                        if is_clique:
                            for id in stk:
                                if id < self.n_agents:
                                    checked[0][id] = 1
                                else:
                                    checked[1][id - self.n_agents] = 1
                            stk = []
                            break
                        u = j
                        visited[j] = True
        
        # find the remove action
        for i in range(2 * self.n_agents):
            u = i
            stk = []
            visited = [False] * (2 * self.n_agents)
            visited[u] = True
            if i < self.n_agents:
                if checked[0][i]:
                    continue
            elif checked[1][i - self.n_agents]:
                continue
            
            for _ in range(2 * self.n_agents):
                for j in range(2 * self.n_agents):
                    if connected_matrix[u][j] == 1 and u != j:
                        congestion = False
                        if j < self.n_agents:
                            x, y = new_pos[0][j]
                            if self.conquer_matrix[1][x][y] == 1:
                                congestion = True
                        else:
                            x, y = new_pos[1][j - self.n_agents]
                            if self.conquer_matrix[0][x][y] == 1:
                                congestion = True
                        if visited[j]:
                            congestion = True
                            
                        visited[j] = True
                        
                        if not congestion:
                            for id in stk:
                                if id < self.n_agents:
                                    checked[0][id] = 1
                                else:
                                    checked[1][id - self.n_agents] = 1
                            stk = []
                            break
                        stk.append(j)
                        u = j
                if len(stk) == 0:
                    break
                                
        for i in range(self.n_agents):
            for j in range(2):
                if checked[j][i]:
                    new_pos[j][i] = dcopy(self.agent_pos[j][i])
        
        
        return new_pos, checked, punish
    
    def step(self, action_1, action_2, render = False):
        
        new_pos, checked, punish = self.get_next_action_pos(action_1, action_2)
        
        # render before action
        for i in range(self.n_agents):
            if checked[0][i] == 0:
                x, y = new_pos[0][i]
                if(self.conquer_matrix[1][x][y] == 0):
                    if self.agent_pos[0][i][0] != new_pos[0][i][0] or self.agent_pos[0][i][1] != new_pos[0][i][1]:
                        self.agents_matrix[0][self.agent_pos[0][i][0]][self.agent_pos[0][i][1]] = 0
                        self.agents_matrix[0][x][y] = 0
                        if(render):
                            self.screen.redraw_squares(self.agent_pos[0][i][0], self.agent_pos[0][i][1], i + 1)
                else:
                    self.agents_matrix[0][x][y] = self.agents_matrix[1][x][y] = 0
                                      
            if checked[1][i] == 0:
                x, y = new_pos[1][i]
                if(self.conquer_matrix[0][x][y] == 0):
                    if self.agent_pos[1][i][0] != new_pos[1][i][0] or self.agent_pos[1][i][1] != new_pos[1][i][1]:
                        self.agents_matrix[1][x][y] = 0
                        self.agents_matrix[1][self.agent_pos[1][i][0]][self.agent_pos[1][i][1]] = 0
                        if(render):
                            self.screen.redraw_squares(self.agent_pos[1][i][0], self.agent_pos[1][i][1], -i - 1)
                else:
                    self.agents_matrix[0][x][y] = self.agents_matrix[1][x][y] = 0
                        
        # render after action
        for i in range(self.n_agents):
            for j in range(2):
                if checked[j][i] == 0:
                    x, y = new_pos[j][i]
                    if(self.conquer_matrix[1 - j][x][y] == 1):
                        self.conquer_matrix[1 - j][x][y] = 0
                        if(render):
                            self.screen.reset_square(x, y, 0)
                        new_pos[j][i] = dcopy(self.agent_pos[j][i])
                    else:
                        self.conquer_matrix[j][x][y] = 1
                        self.agents_matrix[j][x][y] = 1
        
                
        for i in range(self.n_agents):
            self.agent_pos[0][i] = [new_pos[0][i][0], new_pos[0][i][1]]
            self.agent_pos[1][i] = [new_pos[1][i][0], new_pos[1][i][1]]
            
        state = [self.score_matrix, self.agents_matrix, self.conquer_matrix, 
                       self.treasures_matrix, self.walls_matrix]
        common_scores, treasure_scores,  area_matrix = self.compute_score(state)
        self.treasure_score[0] += treasure_scores[0]
        self.treasure_score[1] += treasure_scores[1]
        
        if(render):
            for i in range(self.n_agents):
                self.screen.reset_square(self.agent_pos[0][i][0], self.agent_pos[0][i][1], i + 1)
                self.screen.reset_square(self.agent_pos[1][i][0], self.agent_pos[1][i][1], - i - 1)
            self.screen.show_score()
        
        self.score_mine = common_scores[0] + self.treasure_score[0]
        self.score_opponent = common_scores[1] + self.treasure_score[1]
        reward = self.score_mine - self.score_opponent - self.old_score - punish
        # print(punish, reward)
        self.old_score = reward
        self.remaining_turns -= 1
        if(render):
            self.screen.save_score(self.score_mine, self.score_opponent, self.remaining_turns)
            # print(self.score_mine, self.score_opponent)
        terminal = (self.remaining_turns == 0)
        self.punish += punish/1000
        if not terminal:
            reward = 0
        elif self.score_mine < self.score_opponent:
            reward = -1
        else:
            reward = 1

        return [np.array(flatten(state)), reward, terminal, self.remaining_turns]

    def next_state(self, state, action, player_ID, agent_ID):
        state = dcopy(state)
            
        if agent_ID == self.n_agents - 1:
            player_ID = 1 - player_ID
            agent_ID = 0
        else:
            agent_ID += 1
            
        if player_ID == 1:
            temp = dcopy(state[1][0])
            state[1][0] = dcopy(state[1][1])
            state[1][1] = temp
            temp = dcopy(state[2][0])
            state[2][0] = dcopy(state[2][1])
            state[2][1] = temp
            
        state[-1] = self.get_agent_state(agent_ID, self.agent_pos[player_ID])
        return self.soft_step_(state, action), player_ID, agent_ID
            
    def get_return(self, state, player_ID):
        return self.get_score(state, player_ID)
    
    def is_done_state(self, state, depth):
        return depth >= 2 * (1 + self.n_turns) * self.n_agents
    