import random
from copy import deepcopy as dcopy
import numpy as np
# random.seed(1)            
class Data():
    
    def __init__(self, MIN_SIZE, MAX_SIZE):
        self.MIN_SIZE = MIN_SIZE
        self.MAX_SIZE = MAX_SIZE
    
    def get_random_map(self):
        height = random.randint(self.MIN_SIZE, self.MAX_SIZE)
        # width = random.randint(self.MIN_SIZE, self.MAX_SIZE)
        width = height
        turns = random.randint(30, 60)
        n_agents = random.randint(2, 8)
        n_treasures = random.randint(n_agents, n_agents * 2)
        # n_treasures = 0
        # n_walls =  0
        n_walls =  random.randint(int(height * width / 40), int(height * width / 30))
        score_matrix = []
        conquer_matrix = [[], []]
        mx = random.randint(3, 30)
        matrix = []
        for i in range(height):
            matrix.append([0] * width)
            score_matrix.append([0] * width)
            conquer_matrix[0].append([0] * width)
            conquer_matrix[1].append([0] * width)
            
        for i in range(height):
            for j in range(width):
                value = random.randint(-mx, mx)
                if(value < 0 and random.randint(0, 1) == 0):
                    value = -value
                score_matrix[i][j] =  value
                score_matrix[height- i - 1][width- j - 1] = value
                
        agent_pos = [[], []]
        
        
        for j in range(n_agents):
            _x, _y = random.randint(0, height- 1), random.randint(0, width- 1)
            while  _x == _y or matrix[_x][_y] > 0: 
                _x = random.randint(0, height- 1)
                _y = random.randint(0, width- 1)
            matrix[_x][_y] = 1
            matrix[height- _x - 1][width- _y - 1] = 2
            agent_pos[0]. append([_x, _y])
            agent_pos[1]. append( [height - _x - 1, width - _y - 1])
        
            
        num_walls = random.randint(int(height * width / 40), int(height * width / 30))
        # num_treasures = 0
        treasures = []
        for j in range(n_treasures):
            _x, _y = random.randint(0, height- 1), random.randint(0, width- 1)
            while  _x == _y or matrix[_x][_y] > 0: 
                _x = random.randint(0, height- 1)
                _y = random.randint(0, width- 1)
            # score_matrix[_x][_y] = random.randint(8, 16)
            matrix[_x][_y] = 3
            matrix[height- _x - 1][width- _y - 1] = 3
            # score_matrix[height- _x - 1][width- _y - 1] = random.randint(8, 16)
            value = random.randint(8, 16)
            treasures.append([_x, _y, value])
            treasures.append([height- _x - 1, width- _y - 1, value])
        
               
        num_walls = random.randint(int(height * width / 40), int(height * width / 30))
        # num_walls = random.randint(2, 2)
        # num_walls = 0
        
        wall_coords = []
        for j in range(n_walls):
            _x, _y = random.randint(0, height- 1), random.randint(0, width- 1)
            while  _x == _y or matrix[_x][_y] > 0: 
                _x = random.randint(0, height- 1)
                _y = random.randint(0, width- 1)
            matrix[_x][_y] = 4
            matrix[height- _x - 1][width- _y - 1] = 4
            score_matrix[_x][_y] = 0
            score_matrix[height- _x - 1][width- _y - 1] = 0
            wall_coords.append([_x, _y])
            wall_coords.append([height - _x - 1, width- _y - 1])
        
        data = [height, width, score_matrix, agent_pos, treasures, wall_coords, 
                    conquer_matrix, turns, n_agents]
        return data
                
