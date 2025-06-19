# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math

import numpy as np
from xot.env import Game24


EPS = 1e-8

log = logging.getLogger(__name__)

# 与训练模型阶段的MCTS 基本一致，注释不再赘述
class MCTS():
    def __init__(self, game:Game24, nnet, args, player=1):
        self.game = game
        self.player = player
        self.nnet = nnet
        self.args = args
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.modelCall = 0
    # 获取动作概率分布
    def getActionProb(self, canonicalBoard, temp=1, step=0):
        for i in range(self.args.numMCTSSims):
            self.searchSinglePlayer(canonicalBoard, step=step)
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
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

    def searchSinglePlayer(self, canonicalBoard, step=0):   
        s = self.game.stringRepresentation(canonicalBoard)
        terminate = self.game.isTerminate(canonicalBoard, step)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard) # 存储新状态及其奖励值(0,1,-1)
        if terminate:
            return self.Es[s]
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            self.modelCall += 1 # 神经网络模型的call
            valids = self.game.getValidMoves(canonicalBoard)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            self.Vs[s] = valids
            self.Ns[s] = 0
            return v
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        # UCB选择最佳动作
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.model.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.model.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                if u > cur_best:
                    cur_best = u
                    best_act = a
        a = best_act
        next_s, _ = self.game.getNextState(canonicalBoard, a)
        v = self.searchSinglePlayer(next_s, step+1)
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return v


    def inferSinglePlayer(self, canonicalBoard, step=0, seed=42):
        np.random.seed(seed)
        state = canonicalBoard
        selected_ac_seq = []
        for i in range(self.game.total_game_step):
            terminate = self.game.isTerminate(state, i)
            if terminate:
                break
            s = self.game.stringRepresentation(state)
            # valids = self.game.getValidMoves(state)
            # ac_candidates = [action for (state, action) in self.Qsa.keys() if state == s]
            counts = []
            for a in range(self.game.getActionSize()):
                try:
                    c_ = self.Nsa[(s, a)] 
                    counts.append(c_)
                except:
                    counts.append(0)
            counts_sum = float(sum(counts))
            if counts_sum == 0:
                probs, _ = self.nnet.predict(state)
                counts = probs.tolist()
                counts_sum = float(sum(counts))
                probs = [x / counts_sum for x in counts]
            else:
                probs = [x / counts_sum for x in counts]
            valid_moves = self.game.getValidMoves(state)
            masked_prob = valid_moves * probs
            counts_sum_masked = float(sum(masked_prob))
            probs = [x / counts_sum_masked for x in masked_prob]
            selected_ac = np.random.choice(len(probs), p=probs)
            state, action_in_text= self.game.getNextState(state, selected_ac)
            selected_ac_seq.append(action_in_text)

        res = self.game.getGameEnded(state)
        return selected_ac_seq, res
    
    def getModelCall(self):
        return self.modelCall
    
    def reset(self):
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}