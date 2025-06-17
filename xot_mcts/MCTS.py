# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math

import numpy as np
from scipy.special import softmax

from game24.Game24Game import Game24
from game24.pytorch.NNet import NNetWrapper as NN
EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    def __init__(self, game:Game24, nnet:NN, args, player=1):
        self.game = game
        self.player = player
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # 状态s和动作a对应的Q值
        self.Nsa = {}  # 边(s,a)被访问的次数
        self.Ns = {}  # 状态s被访问的次数
        self.Ps = {}  # 神经网络返回的初始策略 policy
        self.Es = {}  # 游戏是否在状态s结束 getGameEnded
        self.Vs = {}  # 状态s下的合法动作 getValidMoves
        self.modelCall = 0  # 记录调用模型的次数

    # 运行多次 MCTS 模拟，计算每个可能动作的概率分布,36个数字的列表
    def getActionProb(self, canonicalBoard, multi_sol=0, temp=1, step=0):
        # 运行 MCTS 模拟(100次)
        for i in range(self.args.numMCTSSims):
            if self.player == 2:
                self.search(canonicalBoard)
            elif self.player == 1:
                self.searchSinglePlayer(canonicalBoard, step=step)
        s = self.game.stringRepresentation(canonicalBoard)
        # 统计36种动作的访问次数
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        # 完全确定性选择
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        # 引入随机性
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    # 从当前状态[abcd]开始，递归探索，直到叶节点（没走过的状态），然后反向传播
    # 仅返回当前状态的评估值
    def searchSinglePlayer(self, canonicalBoard, step=0):
        # 递归调用直到找到叶节点,并BP到根节点
        s = self.game.stringRepresentation(canonicalBoard) # 获取当前棋盘
        terminate = self.game.isTerminate(canonicalBoard, step) # step用于跟踪游戏状态
        if s not in self.Es:
            # 分配奖励成功1 / 失败-1 / 未结束0
            self.Es[s] = self.game.getGameEnded(canonicalBoard)
        if terminate:
            # 终止节点
            return self.Es[s]
        if s not in self.Ps: # 叶节点未探索，没有策略和价值
            # 预测当前状态的策略(动作概率分布)和评估值(-1~1) [先验概率]
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            self.modelCall += 1 # 记录模型调用次数
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
        # 选择UCB最高的动作
        for a in range(self.game.getActionSize()):
            if valids[a]:
                # Qsa:动作a的平均回报, 初始为0
                if (s, a) in self.Qsa: # 状态s和动作a对应的Q值
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)]) # UCB公式
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS) # Q=0
                if u > cur_best:
                    cur_best = u
                    best_act = a
        a = best_act
        # 执行a，递归到新状态，得到价值v
        next_s, _ = self.game.getNextState(canonicalBoard, a)
        v = self.searchSinglePlayer(next_s, step+1) # 递归到新状态
        # 将价值v沿路径反向传播，更新Q值和访问计数
        if (s, a) in self.Qsa:
            # 如果(s,a)有记录: 增量式更新平均值 (现有平均值*访问次数+ 新值) / (访问次数+1)
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v # 初次赋值
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return v

    # 用已有的 MCTS 统计，直接生成动作序列，不再改树
    def inferSinglePlayer(self, canonicalBoard, step=0, seed=42):
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
