import random
from copy import deepcopy as dcopy
import numpy as np

random.seed(3)
            
class Data():
    
    def __init__(self, MIN_SIZE, MAX_SIZE):
        self.MIN_SIZE = MIN_SIZE
        self.MAX_SIZE = MAX_SIZE
    
    def clip_input(data):
        return
    
    def generate_random_map(self, n_maps, _dir):
        for i in range(n_maps):
            file_name = _dir +  'inp_file_' + str(i) + '.txt'
            f = open(file_name, 'w')
            height = random.randint(self.MIN_SIZE, self.MAX_SIZE)
            width = random.randint(self.MIN_SIZE, self.MAX_SIZE)
                
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
            
            turns = random.randint(30, 70)
            
            # n_agents = random.randint(2, 8)
            n_agents = 1
            
            coord = [0] * (n_agents * 2)
            
            
            for j in range(n_agents):
                _x, _y = random.randint(0, height- 1), random.randint(0, width- 1)
                while  _x == _y or matrix[_x][_y] > 0: 
                    _x = random.randint(0, height- 1)
                    _y = random.randint(0, width- 1)
                matrix[_x][_y] = 1
                matrix[height- _x - 1][width- _y - 1] = 2
                coord[j] = [_x, _y]
                coord[j + n_agents] = [height- _x - 1, width- _y - 1]
            
                
            num_treasures = 1
            treasure_coord = [0] * (num_treasures * 2)
            for j in range(num_treasures):
                _x, _y = random.randint(0, height- 1), random.randint(0, width- 1)
                while  _x == _y or matrix[_x][_y] > 0: 
                    _x = random.randint(0, height- 1)
                    _y = random.randint(0, width- 1)
                # score_matrix[_x][_y] = random.randint(8, 16)
                matrix[_x][_y] = 3
                matrix[height- _x - 1][width- _y - 1] = 3
                # score_matrix[height- _x - 1][width- _y - 1] = random.randint(8, 16)
                value = random.randint(8, 16)
                treasure_coord[j] = [_x, _y, value]
                treasure_coord[j + num_treasures] = [height- _x - 1, width- _y - 1, value]
            
                   
            num_walls = 2
            
            wall_coord = [0] * (num_walls * 2)
            for j in range(num_walls):
                _x, _y = random.randint(0, height- 1), random.randint(0, width- 1)
                while  _x == _y or matrix[_x][_y] > 0: 
                    _x = random.randint(0, height- 1)
                    _y = random.randint(0, width- 1)
                matrix[_x][_y] = 4
                matrix[height- _x - 1][width- _y - 1] = 4
                wall_coord[j] = [_x, _y]
                wall_coord[j + num_walls] = [height - _x - 1, width- _y - 1]
        
            s = str(height) + ' ' + str(width) + '\n'
            f.write(s)
            for i in range(height):
                for j in range(width):
                    f.write(str(score_matrix[i][j]) + ' ')
                f.write('\n')
            
            f.write(str(turns) + '\n')
            f.write(str(n_agents) + '\n')
            for _pair in coord:
                f.write(str(_pair[0]) + ' ' + str(_pair[1]) + '\n')
            
            f.write(str(num_treasures * 2) + '\n')
            for _pair in treasure_coord:
                f.write(str(_pair[0]) + ' ' + str(_pair[1]) + ' ' + str(_pair[2]) + '\n')
            f.write(str(num_walls * 2) + '\n')
            for _pair in wall_coord:
                f.write(str(_pair[0]) + ' ' + str(_pair[1]) + '\n')
    
    def get_random_map(self):
        height = random.randint(self.MIN_SIZE, self.MAX_SIZE)
        width = random.randint(self.MIN_SIZE, self.MAX_SIZE)
            
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
        
        turns = random.randint(5, 15)
        
        # n_agents = random.randint(2, 8)
        n_agents = random.randint(2, self.MAX_SIZE / 2)
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
        
            
        num_treasures = random.randint(2, 4)
        treasures = []
        for j in range(num_treasures):
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
        
               
        # num_walls = random.randint(int(height * width / 40), int(height * width / 30))
        num_walls = random.randint(2, 4)
        
        wall_coords = []
        for j in range(num_walls):
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
    
    def transposion(self, data):
        height, width, score_matrix, agent_pos, treasures, wall_coords, \
                    conquer_matrix, turns, n_agents = dcopy(data)
        
        return [height, width, score_matrix, [agent_pos[1], agent_pos[0]], treasures, wall_coords, 
                    [conquer_matrix[1], conquer_matrix[0]], turns, n_agents]
    
    def read_map_from_file(self, file_name):
        
        data = []
        
        with open(file_name) as f:
            score_matrix = []
            h, w = map(int, f.readline().split())
            for i in range(h):
                array = list(map(int, f.readline().split()))
                while(len(array) < MAX_SIZE):
                    array.append(0)
                score_matrix.append(array)
            while(len(score_matrix) < MAX_SIZE):
                score_matrix.append([0] * MAX_SIZE)
            num_tresures = list(map(int, f.readline().split()))[0]
            treasures = []
            for j in range(num_tresures):
                coord = list(map(int, f.readline().split()))
                treasures.append(coord)
                
            num_walls = list(map(int, f.readline().split()))[0]
            coord_walls = []
            for j in range(num_walls):
                coord = list(map(int, f.readline().split()))
                coord_walls.append(coord)
                
            n_agents = list(map(int, f.readline().split()))[0]
            agent_pos = [[], []]
            for j in range(n_agents * 2):
                coord = list(map(int, f.readline().split()))
                # print(coord)
                if(j < n_agents):
                    agent_pos[0].append(coord)
                else:
                    agent_pos[1].append(coord)
            
            conquer_matrix = [[], []]
            for i in range(h):
                array = list(map(int, f.readline().split()))
                while(len(array) < MAX_SIZE):
                    array.append(0)
                conquer_matrix[0].append(array)
            while(len(conquer_matrix[0]) < MAX_SIZE):
                conquer_matrix[0].append([0] * MAX_SIZE)
            
            conquer_matrix[1] = []
            for i in range(h):
                array = list(map(int, f.readline().split()))
                while(len(array) < MAX_SIZE):
                    array.append(0)
                conquer_matrix[1].append(array)
            while(len(conquer_matrix[1]) < MAX_SIZE):
                conquer_matrix[1].append([0] * MAX_SIZE)
            turns = list(map(int, f.readline().split()))[0]
            data = [h, w, score_matrix, agent_pos, treasures, coord_walls, 
                    conquer_matrix, turns, n_agents]
        return data

    def read_input(self, n_maps, random_map = True):
        data = []
        for i in range(n_maps):
            if random_map:
                data.append(self.get_random_map())
            else:
                file_name = 'Input_File/inp_file_' + str(i) + '.txt'
                data.append(self.read_map_fronm_file(file_name))
        print(data)
        return data
                