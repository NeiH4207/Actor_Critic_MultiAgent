"""
Created on Fri Nov 27 16:00:47 2020
@author: hien
"""
from __future__ import division

import logging
from src.environment import Environment
from read_input import Data
from src.model import Policy
from src.Coach import Coach
from src.utils import dotdict
import torch

log = logging.getLogger(__name__)

args = dotdict({
    'run_mode': 'train',
    'visualize': True,
    'min_size': 7,
    'max_size': 7,
    'n_games': 1,
    'n_maps': 1000,
    'show_screen': True,
    'replay_memory_size': 10000,
    'initial_epsilon': 0.1,
    'final_epsilon': 1e-4,
    'dir': './Models/',
    'numIters': 1000,
    'numEps': 20,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 10000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 5,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'colab_train': False,
    'colab_dir': "/content/drive/MyDrive/trainned_model/agent_mcts.pt",
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('Models','agent_mcts.pt'),
    'numItersForTrainExamplesHistory': 15,
    'saved_model': True
})

def main():
    data = Data(args.min_size, args.max_size)
    env = Environment(data.get_random_map(), args.show_screen, args.max_size)
    model = Policy(env)
    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        model.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    # log.info('Loading the Coach...')
    coach = Coach(env, model, args)

    if args.load_model:
        # log.info("Loading 'trainExamples' from file...")
        coach.loadTrainExamples()

    # log.info('Starting the learning process !')
    coach.learn()


if __name__ == "__main__":
    main()