# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import pandas as pd
import numpy as np
import logging
import coloredlogs

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


"""
Game class implementation for the game of 24.
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Chaoyun Zhang
Date: Sep 28, 2023.
"""

ACTION_SIZE = 36
MASK_VALUE = 0.0001 # 表示“无效”或“已使用”的位置

action_ = ['+','-','in-','*','/','in/'] # 所有可能的动作
action_dic = {0: [0, '+', 1], 1: [0, '-', 1], 2: [1, '-', 0], 3: [0, '*', 1], 4: [0, '/', 1], 5: [1, '/', 0], 6: [0, '+', 2], 7: [0, '-', 2], 
              8: [2, '-', 0], 9: [0, '*', 2], 10: [0, '/', 2], 11: [2, '/', 0], 12: [0, '+', 3], 13: [0, '-', 3], 14: [3, '-', 0], 15: [0, '*', 3], 
              16: [0, '/', 3], 17: [3, '/', 0], 18: [1, '+', 2], 19: [1, '-', 2], 20: [2, '-', 1], 21: [1, '*', 2], 22: [1, '/', 2], 23: [2, '/', 1], 
              24: [1, '+', 3], 25: [1, '-', 3], 26: [3, '-', 1], 27: [1, '*', 3], 28: [1, '/', 3], 29: [3, '/', 1], 30: [2, '+', 3], 31: [2, '-', 3], 
              32: [3, '-', 2], 33: [2, '*', 3], 34: [2, '/', 3], 35: [3, '/', 2]}

class Game24(Game):
    def __init__(self, train_dir='', test_dir='', game_step=3):
        self.action_size = ACTION_SIZE
        self.terminate = False
        self.target = 24
        self.init_board = [8, 8, 5, 5]
        self.train_size = 0
        self.test_size = 0
        self.total_test = 0
        self.n = 4
        self.steps = game_step
        if train_dir: # xot_mcts/game24/data/xxx.csv
            log.info("Loading Training Environment...")
            self.train_data = pd.read_csv("xot_mcts/game24/data/train.csv")
            self.train_size = len(self.train_data)
        if test_dir:
            log.info("Loading Test Environment...")
            self.test_data = pd.read_csv("xot_mcts/game24/data/test.csv")
            self.test_size = len(self.test_data)

    def getInitBoard(self):
        # return initial board (numpy board)
        if self.train_size > 0:
            # 随机抽取指定数量的行作为初始棋盘
            choose = self.train_data['Puzzles'].sample(n=1).values
            b = np.array([int(i) for i in choose[0].split(' ')])
        else:
            b = self.init_board
        return np.array(b)

    def getTestBoard(self):
        # 按顺序从测试集中取出每组数字进行测试
        if self.test_size > 0:
            i = self.total_test % self.test_size
            choose = self.test_data['Puzzles'].iloc[i]
            b = np.array([int(i) for i in choose.split(' ')])
            self.total_test += 1
            return np.array(b)
        return self.getInitBoard()
    
    def TestReset(self):
        self.total_test = 0

    def getBoardSize(self):
        return self.n

    def getActionSize(self):
        # return number of actions
        return self.action_size

    def getNextState(self, board, action):
        # if player takes action on board, return next (board,player)
        # 根据当前棋盘和选择的动作，计算下一步的状态
        # 从 action_dic 中获取动作对应的 [num1, op, num2]
        action_value = action_dic[action]
        num1, operator, num2 = action_value
        step = board.tolist().count(MASK_VALUE)
        step += 1
        expression = str(board[num1]) + str(operator) + str(board[num2]) 
        try:
            result = eval(expression)
        except:
            result = float("inf")
        # 移除被合并的数字
        remaining = [x for i, x in enumerate(board) if i not in [num1, num2] and x != MASK_VALUE]
        n1, n2 = board[num1], board[num2] # 需要被合并的数字
        exp_in_text = [str(operator), int(n1) if int(n1)==n1 else n1, int(n2) if int(n2)==n2 else n2]
        # 合并结果与剩余数字, 再加一个MASK_VALUE --> [1, 3, 8, MASK]排序后
        next_state = sorted([result] + remaining) + [MASK_VALUE] * step
        return np.array(next_state), exp_in_text # 下一个状态 和 当前操作文本描述
        
        
    # 只能对未被使用的两个数字进行运算 不能除0 
    def getValidMoves(self, board):
        # 长度为 36 的[01]数组，表示每个动作是否有效
        valids = np.ones(ACTION_SIZE, dtype=np.float32)
        step = board.tolist().count(MASK_VALUE)
        valid_index = self.getBoardSize() - step # 4-0
        count = 0
        for num1 in range(self.getBoardSize()):
            for num2 in range(num1 + 1,self.getBoardSize()):
                for op in action_:
                    operator = op[-1]
                    if 'in' in op:
                        expression_list = [num2, operator, num1]
                    else:
                        expression_list = [num1, operator, num2]
                    if expression_list[0] > valid_index - 1 or expression_list[2] > valid_index - 1:
                        valids[count] = 0
                    elif board[expression_list[2]] == 0 and expression_list[1] == '/':
                        valids[count] = 0
                    count += 1
        return valids


    def getGameEnded(self, board):
        # 成功1 / 失败-1 / 未结束0
        terminate = board.tolist().count(MASK_VALUE) >= 3
        if terminate:
            if np.abs(board[0] - self.target) < 1e-4:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0
        return reward
    

    def isTerminate(self, board, step):
        # [1, 5, 9, 9], 0
        return step >= self.steps or board.tolist().count(MASK_VALUE) >= 3 # 已经超过步数或者填满了
      

    def getCanonicalForm(self, board, player):
        # 获取规范形式
        return player*board

    def getSymmetries(self, board, pi):
        # 棋盘不对称
        return [(board, pi)]

    def stringRepresentation(self, board):
        return tuple(board.tolist())
