# MODULES
import pygame, sys
import numpy as np
import random
import time
from copy import deepcopy as dcopy

# initializes pygame

# ---------
# CONSTANTS
# ---------

# rgb: red green blue
RED = (255, 0, 0)
BG_COLOR = (231, 225, 232)
LINE_COLOR = (0, 0, 0)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)
    
class Screen():
    def __init__(self, env):
        if env.show_screen:
            pygame.init()
            self.env = env
            self.WIDTH = 800
            self.HEIGHT = 800
            self.LINE_WIDTH = 1
            self.SQUARE_SIZE = int(self.HEIGHT / 20)
            self.color_A = (255, 172,  88)
            self.color_B = (129, 188, 255)
            if env.show_screen:
                self.load_image()
                pygame.display.set_caption( 'ProCon-2020' ) 

    def render(self):
        pygame.display.update()

    def load_image(self):
        self.agent_A_img = pygame.transform.scale(
            pygame.image.load('GameBoard/images/agent1.png'), (self.SQUARE_SIZE, self.SQUARE_SIZE))
        self.agent_B_img = pygame.transform.scale(
            pygame.image.load('GameBoard/images/agent2.png'), (self.SQUARE_SIZE, self.SQUARE_SIZE))
        self.wall_img =  pygame.transform.scale(
            pygame.image.load('GameBoard/images/wall.jpg'), (self.SQUARE_SIZE, self.SQUARE_SIZE))
        self.background_img = pygame.transform.scale(
            pygame.image.load('GameBoard/images/background.jpg'), (626, 966))
        self.table_img =  pygame.transform.scale(
            pygame.image.load('GameBoard/images/board.png'), (400, 350))
        self.treasure_img = pygame.transform.scale(
            pygame.image.load('GameBoard/images/treasure.jpg'),
            (int(self.SQUARE_SIZE / 2), int(self.SQUARE_SIZE / 2)))  
        
    def coord(self, x, y):
        return x * self.SQUARE_SIZE, y * self.SQUARE_SIZE
    
    def setup(self, env): 
        self.h = env.height
        self.w = env.width
        self.screen = pygame.display.set_mode(self.coord(self.h + 8, self.w))  
        self.screen.fill( BG_COLOR )
        self.draw_lines()
        self.screen.blit(self.background_img, self.coord(self.h, 0))
        for i in range(self.h):
            for j in range(self.w):
                # draw wall
                if self.env.wall_board[i][j] == 1:
                    self.draw_wall(i, j)
                else:
                    self.reset_square([i, j], -1)
                    
                # draw treasure
                if self.env.treasure_board[i][j] > 0:
                    self.show_treasure_value(self.env.norm_treasure_board[i][j], i, j)
                    
                # draw agent
        for player_id in range(self.env.num_players):
            for agent_ID in range(self.env.n_agents):
                self.draw_squares(self.env.agent_pos[player_id][agent_ID], player_id)
                self.reset_square(self.env.agent_pos[player_id][agent_ID], player_id, agent_ID)
                    
        self.show_score()
        pygame.display.update()

    def start(self):
        game_over = False
        # -------
        
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                    sys.exit()
                    # mouseX = event.pos[0] # x
                    # mouseY = event.pos[1] # y
                    
                    # clicked_row = int(mouseY // self.SQUARE_SIZE)
                    # clicked_col = int(mouseX // self.SQUARE_SIZE)
        
        
            pygame.display.update()
            
    def show_score(self):
        self.screen.blit(self.table_img, self.coord(self.h - 1, -2))
        
        myFont = pygame.font.SysFont("Times New Roman", 30)
        
        color = (255, 178, 21)
        
        SA = myFont.render("    : " + str(round(self.env.players[0].total_score)), 0, color)
        SB = myFont.render("    : " + str(round(self.env.players[1].total_score)), 0, color)
        STurns = myFont.render("Turns: " + str(self.env.remaining_turns), 0, color)
        
    
        self.screen.blit(SA, self.coord(self.h + 1, 1))
        self.screen.blit(SB, self.coord(self.h + 1, 2))
        self.screen.blit(STurns, self.coord(self.h + 1, 3))
        self.screen.blit(self.agent_A_img, (self.h * self.SQUARE_SIZE + 30, -5 + 1 * self.SQUARE_SIZE))
        self.screen.blit(self.agent_B_img, (self.h  * self.SQUARE_SIZE+ 30, -5 + 2 * self.SQUARE_SIZE))
    
    def show_value(self, value, x, y):
        myFont = pygame.font.SysFont("Times New Roman", 20)
        value = round(value)
        pos = 5
        if value >= 0 and value < 10:
            pos = 15
        elif value > 10 or value > -10:
            pos = 10
        value = myFont.render(str(value), 1, (0, 0, 0))
        self.screen.blit(value, (x * self.SQUARE_SIZE + pos, y * self.SQUARE_SIZE + 8))
        
    def show_index_agent(self, x, y, agent_ID):
        myFont = pygame.font.SysFont("Times New Roman", 13)
        agent_ID = myFont.render(str(abs(agent_ID)), 1, (0, 111, 220))
        self.screen.blit(agent_ID, (x * self.SQUARE_SIZE + 17, y * self.SQUARE_SIZE + 20))
        
        
    def show_treasure_value(self, value, x, y):
        self.draw_treasure(x, y)
        value = round(value)
        myFont = pygame.font.SysFont("Times New Roman", 13)
        value = myFont.render(str(value), 1, (255, 0, 0))
        self.screen.blit(value, (x * self.SQUARE_SIZE + 2, y * self.SQUARE_SIZE + int(self.SQUARE_SIZE * 5 / 7)))
        
    def draw_wall(self, x, y):
        self.screen.blit(self.wall_img, self.coord(x, y))
        
    def draw_treasure(self, x, y):
        self.screen.blit(self.treasure_img, self.coord(x, y))
        
    def draw_agent(self, x, y, player_ID, agent_ID):
        player_img = self.agent_A_img if player_ID == 0 else self.agent_B_img
        self.screen.blit(player_img, self.coord(x, y))
        self.show_index_agent(x, y, agent_ID + 1)
        
    
    def draw_lines(self):
        for i in range(self.w):
            pygame.draw.line(self.screen, LINE_COLOR, (0, i * self.SQUARE_SIZE), 
                              (self.h * self.SQUARE_SIZE, i * self.SQUARE_SIZE), self.LINE_WIDTH )
        for i in range(self.h):
            pygame.draw.line(self.screen, LINE_COLOR, (i * self.SQUARE_SIZE, 0),
                             (i * self.SQUARE_SIZE, self.w * self.SQUARE_SIZE), self.LINE_WIDTH )
    
    def _draw_squares(self, x1, y1, x2, y2, player_ID):
        color = self.color_A if player_ID == 0 else self.color_B
        pygame.draw.rect(self.screen, color, (x1, y1, x2, y2))
        
        
    def draw_squares(self, coord, player_ID):
        x, y = coord
        self._draw_squares(2 + x * self.SQUARE_SIZE, 2 + y * self.SQUARE_SIZE,
                           (self.SQUARE_SIZE - 3), (self.SQUARE_SIZE - 3), player_ID)
        
    def _redraw_squares(self, x1, y1, x2, y2, player_ID):
        color = self.color_A if player_ID == 0 else self.color_B
        pygame.draw.rect(self.screen, color, (x1, y1, x2, y2))
        
        
    def redraw_squares(self, x, y, player_ID):
        self._redraw_squares(2 + x * self.SQUARE_SIZE, 2 + y * self.SQUARE_SIZE,
                           (self.SQUARE_SIZE - 3), (self.SQUARE_SIZE - 3), player_ID)
        self.show_value(self.env.norm_score_board[x][y], x, y)
           
    def _reset_squares(self, x1, y1, x2, y2, player_ID):
        color = self.color_A if player_ID == 0 else self.color_B
        if player_ID < 0:
            color = BG_COLOR
        pygame.draw.rect(self.screen, color, (x1, y1, x2, y2))
        
    def reset_square(self, coord, player_ID, agent_ID = 0):
        x, y = coord
        self._reset_squares(2 + x * self.SQUARE_SIZE, 2 + y * self.SQUARE_SIZE,
                           (self.SQUARE_SIZE - 3), (self.SQUARE_SIZE - 3), player_ID)
        if player_ID >= 0:
            self.draw_agent(x, y, player_ID, agent_ID)
        else:
            self.show_value(self.env.norm_score_board[x][y], x, y)
        
    def reset(self):
        self.screen.fill( BG_COLOR )
        # self.draw_lines()
        self.screen.blit(self.background_img, self.coord(self.h, 0))
        # self.screen.blit(self.background_img, self.coord(20, 0))
        self.draw_lines()
        for i in range(self.h):
            for j in range(self.w):
                # draw wall
                if self.env.wall_board[i][j] == 1:
                    self.draw_wall(i, j)
                else:
                    self.reset_square([i, j], -1, 0)
                    
                # draw treasure
                if self.env.treasure_board[i][j] > 0:
                    self.show_treasure_value(self.env.norm_treasure_board[i][j], i, j)
                    
                # draw agent
        for player_id in range(self.env.num_players):
            for agent_ID in range(self.env.n_agents):
                self.draw_squares(self.env.agent_pos[player_id][agent_ID], player_id)
                self.reset_square(self.env.agent_pos[player_id][agent_ID], player_id, agent_ID)
                    
        self.show_score()