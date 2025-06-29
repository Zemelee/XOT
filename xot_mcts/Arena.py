import logging
import numpy as np
from tqdm import tqdm

from game24.Game24Game import Game24
log = logging.getLogger(__name__)

# 竞技场类：用于模拟两个智能体之间的对战（单步决策）
class ArenaSingle():
    def __init__(self, mcts1, mcts2, game:Game24, winReward=1):
        # 用于比较两个MCTS玩家的表现
        self.mcts1 = mcts1 # 旧
        self.mcts2 = mcts2 # 新
        # 定义两个玩家的策略函数，使用MCTS获取最佳动作
        self.player1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0, step=0))
        self.player2 = lambda x: np.argmax(mcts2.getActionProb(x, temp=0, step=0))
        self.game = game
        self.winReward = winReward # 胜利奖励值，判断是否胜利

    def playGame(self, player, verbose=False):
        # 模拟单次游戏
        # player: 当前玩家的策略函数 verbose: 是否输出详细信息
        if self.game.test_size > 0:
            board = self.game.getTestBoard()  # 获取测试棋盘状态
        else:
            board = self.game.getInitBoard()  # 获取初始棋盘状态
        step = 0
        while not self.game.isTerminate(board, step):  # 检查是否游戏结束
            action = player(board)  # 玩家选择动作
            valids = self.game.getValidMoves(board)  # 获取所有合法动作
            if valids[action] == 0:
                log.error(f'动作 {action} 不合法！')
                assert valids[action] > 0  # 验证动作合法性
            board, _ = self.game.getNextState(board, action)  # 获取下一个状态
            step += 1
        if verbose:
            print("结果 ", str(self.game.getGameEnded(board)))
        return self.game.getGameEnded(board)  # 返回游戏结果

    def playGames(self, num, verbose=False):
        oneWon = 0
        twoWon = 0
        self.game.TestReset()  # 测试重置
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            self.mcts1.reset()
            gameResult = self.playGame(self.player1, verbose=verbose)
            if gameResult == self.winReward:
                oneWon += 1  # player1胜利计数
        self.game.TestReset()
        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            self.mcts2.reset()
            gameResult = self.playGame(self.player2, verbose=verbose)
            if gameResult == self.winReward:
                twoWon += 1  # player2胜利计数
        self.game.TestReset()
        return oneWon, twoWon


# 测试用竞技场类，用于评估单个玩家在多个问题上的表现
class ArenaTest():
    def __init__(self, mcts1, game:Game24, multi_sol=0, winReward=1):
        self.mcts1 = mcts1
        self.player1 = lambda x: np.argmax(self.mcts1.getActionProb(x, multi_sol, temp=0, step=0))
        self.game = game
        self.winReward = winReward
        self.multi_sol = multi_sol

    def playGame(self, player, verbose=False):
        if self.game.test_size > 0:
            board = self.game.getTestBoard()
        else:
            board = self.game.getInitBoard()
        # it = 0
        problem_state = board  # 记录初始状态
        step = 0
        actions = []  # 动作序列
        while not self.game.isTerminate(board, step):
            action = player(board) # 根据策略选择概率最高的动作索引
            valids = self.game.getValidMoves(board)
            assert valids[action] > 0
            board, action_in_text = self.game.getNextState(board, action)
            actions.append(action_in_text)
            step += 1
            if str(self.game.getGameEnded(board)) != "0":
                print("game over: ", str(self.game.getGameEnded(board)))
        return problem_state, self.game.getGameEnded(board), actions

    def playGames(self, num, multi_times, verbose=False):
        # 执行num次游戏
        oneWon = 0
        thoughts_record = []
        self.game.TestReset()
        for i in range(num): # 50
            print(f'第 {i+1}/{num} 局, ', end=" ")
            self.game.total_test = i + 1
            self.mcts1.reset()
            # 初始状态，游戏结果，动作列表 self.player1即nn预测动作概率分布的方法
            problem_state, gameResult, actions = self.playGame(self.player1, verbose=verbose)
            thoughts_record.append([str(problem_state), str(actions), gameResult == self.winReward])
            if self.multi_sol:
                for sol in range(multi_times):
                    selected_ac_seq, res = self.mcts1.inferSinglePlayer(problem_state, step=0, seed=sol)
                    if selected_ac_seq is not None:
                        thoughts_record.append([str(problem_state), str(selected_ac_seq), res == self.winReward])
            if gameResult == self.winReward:
                oneWon += 1
        self.game.TestReset()
        return oneWon, thoughts_record