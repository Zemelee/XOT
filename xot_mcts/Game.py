# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class Game():
    """
    该类定义了游戏的基类。要定义你自己的游戏，继承此类并实现以下函数。
    此类适用于双人对抗性回合制游戏。
    使用 1 表示玩家1，-1 表示玩家2。
    参见 othello/OthelloGame.py 获取具体实现示例。
    """
    def __init__(self):
        pass

    def getInitBoard(self):
        """
        返回：
            startBoard: 对棋盘的一种表示形式（理想情况下是输入到你的神经网络时所使用的格式）
        """
        pass

    def getBoardSize(self):
        """
        返回(x,y): 一个表示棋盘尺寸的元组
        """
        pass

    def getActionSize(self):
        """
        返回：
            actionSize: 所有可能动作的总数
        """
        pass

    def getNextState(self, board, player, action):
        """
        输入：
            board: 当前棋盘状态
            player: 当前玩家（1 或 -1）
            action: 当前玩家执行的动作

        返回：
            nextBoard: 应用动作之后的新棋盘状态
            nextPlayer: 下一步轮到的玩家（应为 -player）
        """
        pass

    def getValidMoves(self, board, player):
        """
        输入：
            board: 当前棋盘状态
            player: 当前玩家

        返回：
            validMoves: 一个长度为 self.getActionSize() 的二进制向量，
                        1 表示当前棋盘和玩家下合法的动作位置，0 表示非法动作
        """
        pass

    def getGameEnded(self, board, player):
        """
        输入：
            board: 当前棋盘状态
            player: 当前玩家（1 或 -1）

        返回：
            r: 如果游戏未结束返回 0。如果玩家胜利返回 1，失败返回 -1，
               平局则返回一个小的非零值。
        """
        pass

    def getCanonicalForm(self, board, player):
        """
        输入：
            board: 当前棋盘状态
            player: 当前玩家（1 或 -1）

        返回：
            canonicalBoard: 返回棋盘的标准形式。标准形式应该与玩家无关。
                            例如在国际象棋中，标准形式可以设定为始终以白方视角表示棋盘。
                            当前玩家是白方时，直接返回原棋盘；
                            当前玩家是黑方时，可以翻转颜色后返回。
        """
        pass

    def getSymmetries(self, board, pi):
        """
        输入：
            board: 当前棋盘状态
            pi: 策略向量，大小为 self.getActionSize()

        返回：
            symmForms: 一个 [(board,pi)] 形式的列表，其中每个元组都是棋盘及其对应的策略向量的一个对称形式。
                       在使用示例训练神经网络时会用到这个函数。
        """
        pass

    def stringRepresentation(self, board):
        """
        输入：
            board: 当前棋盘状态

        返回：
            boardString: 将棋盘快速转换为字符串格式。
                         MCTS 中用于哈希操作。
        """
        pass