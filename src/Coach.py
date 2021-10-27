import logging
import os
import sys
from read_input import Data
from collections import deque
from src.environment import Environment
from pickle import Pickler, Unpickler
from random import shuffle
from read_input import Data
from copy import deepcopy as dcopy
import time
import numpy as np
from tqdm import tqdm
from src.utils import AverageMeter2
import matplotlib.pyplot as plt
# from Arena import Arena
from src.MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        # self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.scores = AverageMeter2()
        
    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.get_observation(0)
        # self.game.log_state(board)
        episodeStep = 0
        agent_pos, player_id, agent_id, depth = self.game.get_agent_pos_all(), 0, 0, 0
        
        self.mcts = MCTS(self.game, self.nnet, self.args, agent_pos, 
                         player_id, agent_id, depth)  # reset search tree

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold * self.game.n_turns)
            pi = self.mcts.getActionProb(board, agent_pos, player_id, agent_id, depth, temp=temp)
            _state, _agent_pos = dcopy(board), dcopy(agent_pos)
            
            if player_id == 1:
                self.game.convert_to_opn_obs(_state, _agent_pos)
            
            agent_step = self.game.get_agent_for_step(agent_id, player_id, _agent_pos)
            
            state_sym, agent_step_sym = \
                self.game.get_symmetric_state(self.game.get_states_for_step(_state),
                                              agent_step)
            
            trainExamples.append([state_sym, player_id, pi, agent_step_sym])

            action = np.random.choice(len(pi), p=pi)
            
            # print(player_id, action)
            board, agent_pos, player_id, agent_id, depth = \
                self.game.get_next_state(board, action, agent_pos, 
                                         player_id, agent_id, depth,
                                         render = self.args.show_screen)
            
            # print(agent_pos, player_id, agent_id, depth)
            r = self.game.getGameEnded(board, player_id, agent_id, depth)
            if r != 0:
                self.scores.update(self.game.players[0].total_score,
                                   self.game.players[1].total_score)
                self.game.soft_reset()
                data = Data(self.args.min_size, self.args.max_size)
                self.game.__init__(data.get_random_map(), self.args.show_screen, self.args.max_size)
                # print([r * ((-1) ** (x[1] != player_id)) for x in trainExamples])
                return [(x[0], x[3], x[2], r * ((-1) ** (x[1] != player_id))) for x in trainExamples]

    def learn(self, iter):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        # examples of the iteration
        if not self.skipFirstSelfPlay or iter > 1:
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                iterationTrainExamples += self.executeEpisode()
                
            if self.args.visualize:
                self.scores.plot()

            # save the iteration examples to the history 
            self.trainExamplesHistory.append(iterationTrainExamples)

        if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            log.warning(
                f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
            self.trainExamplesHistory.pop(0)
            
        # backup history to a file
        self.saveTrainExamples()

        # shuffle examples before training
        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)
        self.nnet.train_examples(trainExamples)
        
        if self.args.visualize:
            plt.rcParams["figure.figsize"] = (20,5)
            plt.subplot(1, 2, 1)
            plt.title('PolicyLoss')
            plt.plot(list(self.nnet.pi_losses.mean_vals)[3:], 'r')
            plt.subplot(1, 2, 2)
            plt.title('ValueLoss')
            plt.plot(list(self.nnet.v_losses.mean_vals)[3:], 'b')
            plt.show()
        # training new network, keeping a copy of the old one
        self.nnet.save_checkpoint(folder=self.args.load_folder_file[0], 
                                  filename=self.args.load_folder_file[1])
        if self.args.colab_train:
            self.nnet.save_colab_model(self.args.colab_dir)

        
    def getCheckpointFile(self):
        return 'checkpoint_' + 'pt'

    def saveTrainExamples(self):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile() + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed
        print('\nTrain Examples saved succesful!')

    def loadTrainExamples(self):
        folder = self.args.checkpoint
        modelFile = os.path.join(folder, self.getCheckpointFile())
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
