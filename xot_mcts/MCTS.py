# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math

import numpy as np
from scipy.special import softmax

from game24.Game24Game import Game24
EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    def __init__(self, game:Game24, nnet, args, player=1):
        self.game = game
        self.player = player
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # 存储状态s和动作a对应的Q值
        self.Nsa = {}  # 存储边(s,a)被访问的次数
        self.Ns = {}  # 存储状态s被访问的次数
        self.Ps = {}  # 存储神经网络返回的初始策略 policy
        self.Es = {}  # 存储游戏是否在状态s结束 getGameEnded
        self.Vs = {}  # 存储状态s下的合法动作 getValidMoves
        self.modelCall = 0  # 记录调用模型的次数

    def getActionProb(self, canonicalBoard, multi_sol=0, temp=1, step=0):
        # 该函数从 canonicalBoard([3,6,10,11]) 开始执行 numMCTSSims 次 MCTS 模拟。
        # 返回一个策略向量，其中第i个动作的概率正比于 Nsa[(s,a)]**(1./temp)
        for i in range(self.args.numMCTSSims):
            if self.player == 2:
                self.search(canonicalBoard)
            elif self.player == 1:
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
        """
        该函数执行一次 MCTS 迭代。它会递归调用直到找到叶节点。
        每个节点选择的动作是具有最大上置信界（UCB）的那个。
        找到叶节点后，调用神经网络返回该状态的初始策略 P 和评估值 v。
        此值将沿着搜索路径反向传播。如果叶节点是终止状态，
        则结果也将沿路径反向传播。同时更新 Ns、Nsa、Qsa 的值。
        注意：返回值是当前状态值的负数。这是因为 v ∈ [-1,1]，
              如果 v 是当前玩家的状态值，则对另一玩家来说是 -v。
        返回：
            v: 当前 canonicalBoard 状态值的负数
        """
        s = self.game.stringRepresentation(canonicalBoard) # 获取棋盘当前binary表示
        terminate = self.game.isTerminate(canonicalBoard, step) # 判断是否终止
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard)
        if terminate:
            # 终止节点
            return self.Es[s]
        if s not in self.Ps: # 叶节点
            # 预测当前状态的策略(动作概率分布)和评估值(即胜负)
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            self.modelCall += 1
            valids = self.game.getValidMoves(canonicalBoard)
            self.Ps[s] = self.Ps[s] * valids  # 给每个动作：屏蔽无效动作
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:# 归一化动作概率
                self.Ps[s] /= sum_Ps_s
            else:
                # 如果所有有效动作都被掩码(全都无效)，则让所有有效动作概率相等
                log.error("所有有效动作均被掩码，正在执行补救措施。")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            
            self.Vs[s] = valids # Vs:状态s下的合法动作
            self.Ns[s] = 0 # Ns:状态s被访问的次数
            return v
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        # 选择上置信界最高的动作
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa: # 状态s和动作a对应的Q值
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)]) # UCB公式
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS) # Q=0
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
        """
        该函数用于推理阶段执行 MCTS，生成完整的动作序列。
        与 search 不同，该函数不会修改树结构，而是根据已有的统计信息进行选择。
        返回：
            selected_ac_seq: 动作序列
            res: 最终游戏结果
        """
        np.random.seed(seed)
        state = canonicalBoard
        selected_ac_seq = []
        for i in range(self.game.steps):
            terminate = self.game.isTerminate(state, i)
            if terminate:
                break
            s = self.game.stringRepresentation(state)
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
                probs = probs.tolist()
            else:
                probs = [x / counts_sum for x in counts]
         
            valid_moves = self.game.getValidMoves(state)
            masked_prob = valid_moves * probs
            counts_sum_masked = float(sum(masked_prob))
            probs = [x / counts_sum_masked for x in masked_prob]

            selected_ac = np.random.choice(len(probs), p=probs)
            state, action_in_text = self.game.getNextState(state, selected_ac)
            selected_ac_seq.append(action_in_text)
        res = self.game.getGameEnded(state)
        return selected_ac_seq, res

    def getModelCall(self):
        # 返回调用神经网络模型的总次数。
        return self.modelCall

    def reset(self):
        # 重置 MCTS 树，清空所有统计信息。
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}