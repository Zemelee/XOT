# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import pandas as pd
import numpy as np
from tqdm import tqdm

from Arena import ArenaSingle, ArenaTest
from MCTS import MCTS

class Coach():
    # 本类执行自我对弈 + 学习过程。它使用 Game 和 NeuralNet 中定义的函数。
    # 参数 args 在 main.py 中指定。

    def __init__(self, game, nnet, args, player=2):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # 竞争者网络(旧网络)
        self.args = args
        self.player = player
        self.mcts = MCTS(self.game, self.nnet, self.args, self.player)
        self.trainExamplesHistory = []  # 存储多个迭代中的自我对弈样本
        self.skipFirstSelfPlay = False  # 可以在loadTrainExamples()中重写
        self.multi_sol = args.multi_sol
        self.multi_times = args.multi_times

    def executeEpisode(self):
         # 本函数执行一次自我对弈回合，从玩家1开始。
         # 游戏进行过程中，每一步都会作为一个训练样例加入到 trainExamples 中。
         # 游戏结束后，根据游戏结果给每个样例分配值。
         # 如果 episodeStep < tempThreshold 使用 temp=1，之后使用 temp=0。
         # 返回：
         #     trainExamples: 示例列表，形式为 (canonicalBoard, currPlayer, pi, v)
         #         pi 是 MCTS 得出的策略向量，v 是如果玩家最终胜利则为 +1，
         #         否则为 -1。
        trainExamples = []
        board = self.game.getInitBoard() # [1, 3, 4, 5]
        self.curPlayer = 1 # 当前玩家初始化为 1
        episodeStep = 0 # 当前回合步数
        rewards = [0] # 奖励列表，初始为 0
        # 进行游戏
        while True:            
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer) if self.player == 2 else board
            temp = int(episodeStep < self.args.tempThreshold) # 早期保留探索性(1) 后期选择 MCTS 中最高概率的动作
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp, step=episodeStep)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])
            action = np.random.choice(len(pi), p=pi)
            if self.player == 2:
                board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
                r = self.game.getGameEnded(board, self.curPlayer)
            else:
                board, self.curPlayer = self.game.getNextState(board, action)
                r = self.game.getGameEnded(board)
            rewards.append(r)
            episodeStep += 1
            terminate = self.game.isTerminate(board, episodeStep)
            if terminate:
                sym = self.game.getSymmetries(board, pi)
                for b, p in sym:
                    trainExamples.append([b, self.curPlayer, p, None])
                return [(x[0], x[2], sum(rewards[i:])) for i, x in enumerate(trainExamples)]


    def learn(self):
        # 执行 numIters(3) 次迭代，每次迭代包含 numEps 场自我对弈。
        # 每次迭代后，使用 trainExamples 中的样例重新训练神经网络（最大长度 maxlenOfQueue）。
        # 然后将新神经网络与旧版本对战，只有当胜率 >= updateThreshold 时才接受新模型。
        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            logging.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args, self.player)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()
                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                logging.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # 备份历史记录到文件
            # 注意：这些样例是使用上一轮模型收集的，因此是 i-1
            self.saveTrainExamples(i - 1)
            # 训练前打乱数据
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            # 训练新网络，并保存旧网络副本
            self.nnet.save_checkpoint(folder=self.args.checkpoint + self.args.env + '/', filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint + self.args.env + '/', filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args, self.player)
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args, self.player)
            logging.info('PITTING AGAINST PREVIOUS VERSION')
            pmcts_modelcall_before = pmcts.getModelCall()
            nmcts_modelcall_before = nmcts.getModelCall()
            arena = ArenaSingle(pmcts, nmcts, self.game, self.args.winReward)
            pwins, nwins = arena.playGames(self.args.arenaCompare, verbose=True)
            pmcts_modelcall_after = pmcts.getModelCall()
            nmcts_modelcall_after = nmcts.getModelCall()

            pmcts_modelcall_avg = round((pmcts_modelcall_after - pmcts_modelcall_before) / self.args.arenaCompare, 2)
            nmcts_modelcall_avg = round((nmcts_modelcall_after - nmcts_modelcall_before) / self.args.arenaCompare, 2)

            logging.info('NEW/PREV WINS : %d / %d, NEW/PREV AVG CALL : %s / %s, ' % (nwins, pwins, nmcts_modelcall_avg, pmcts_modelcall_avg))

            if pwins + nwins == 0 or float(nwins - pwins) / self.args.arenaCompare < self.args.updateThreshold:
                logging.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint + self.args.env + '/', filename='temp.pth.tar')
            else:
                logging.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint + self.args.env + '/', filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint + self.args.env + '/', filename='best.pth.tar')


    def infer(self):
        # 加载最佳模型
        self.pnet.load_checkpoint(folder=self.args.checkpoint + self.args.env + '/', filename='best.pth.tar')
        pmcts = MCTS(self.game, self.pnet, self.args, self.player)

        logging.info('TESTING BEGAIN:')
        
        pmcts_modelcall_before = pmcts.getModelCall()
   
        arena = ArenaTest(pmcts, self.game, self.multi_sol, self.args.winReward)
        pwins, thoughts_record = arena.playGames(self.args.arenaCompare, self.multi_times, verbose=True)
        pmcts_modelcall_after = pmcts.getModelCall()

        pmcts_modelcall_avg = round((pmcts_modelcall_after - pmcts_modelcall_before) / self.args.arenaCompare, 2)
        thoughts_acc = round(pwins/self.game.test_size, 4) * 100

        logging.info('TESTING WINS :  %d / %d, THOUGHTS ACC : %d %%, TESTING AVG CALL : %s' % (pwins, self.game.test_size, thoughts_acc, pmcts_modelcall_avg))
        pd_thoughts = pd.DataFrame(data=thoughts_record, columns=['problem_state', 'thoughts', 'acc'])
        pd_thoughts.to_csv('./logs/%s_thoughts.csv'%self.args.env)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint + self.args.env + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            logging.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            logging.info("找到训练样例文件，正在加载...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            logging.info('加载完成!')

            # 这些样例是基于已有模型收集的，跳过首次自我对弈
            self.skipFirstSelfPlay = True
