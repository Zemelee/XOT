import logging
from argparse import ArgumentParser
from Coach import Coach

from utils import *


def main():
    parser = ArgumentParser("XOT!")
    parser.add_argument('--env', type=str, default='game24')
    parser.add_argument('--mode', type=str, default='train') # train or test
    parser.add_argument('--numIters', type=int, default=3) # Number of iteration.
    parser.add_argument('--numEps', type=int, default=10)  # Number of complete self-play games to simulate during a new iteration.
    parser.add_argument('--updateThreshold', type=float, default=0) # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    parser.add_argument('--maxlenOfQueue', type=int, default=10000) # Number of game examples to train the neural networks.
    parser.add_argument('--numMCTSSims', type=int, default=2000) # MCTS模拟次数.
    parser.add_argument('--tempThreshold', type=int, default=15) # 温度阈值.
    parser.add_argument('--arenaCompare', type=int, default=100) # Number of games to play during arena play to determine if new net will be accepted.
    parser.add_argument('--cpuct', type=float, default=1)
    parser.add_argument('--winReward', type=float, default=1)
    parser.add_argument('--checkpoint', type=str, default='./temp/')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--load_folder_file', type=tuple, default=('/dev/models','best.pth.tar'))
    parser.add_argument('--numItersForTrainExamplesHistory', type=int, default=1000)
    parser.add_argument('--training_env', type=str, default='')
    parser.add_argument('--test_env', type=str, default='game24/data/test.csv')
    parser.add_argument('--multi_sol', type=int, default=1)
    parser.add_argument('--multi_times', type=int, default=50) # 测试模式下的尝试次数
    args = parser.parse_args()

    logging.basicConfig(filename='logs/%s_%s.log'%(args.env, args.mode), filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
    logging.info(args)

    if args.env.lower() == 'game24':
        from game24.Game24Game import Game24 as Game
        from game24.pytorch.NNet import NNetWrapper as nn
    else:
        raise ValueError
    logging.info(f'正在加载 {Game.__name__}...', )
    g = Game(args.training_env, args.test_env) # 加载游戏动作实例，棋盘等
    logging.info(f'正在加载 {nn.__name__}...', )
    nnet = nn(g) # 加载辅助网络
    if args.mode.lower() == 'train':
        if args.load_model:
            logging.info(f'正在加载检查点 "{args.load_folder_file[0]}/{args.load_folder_file[1]}"...')
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        else:
            logging.warning('未加载检查点!')
        logging.info('正在加载Coach类...')
        c = Coach(g, nnet, args, player=1) # c.mcts.getActionProb(mcts.searchSinglePlayer)
        if args.load_model:
            logging.info("正在从文件中加载trainExamples...")
            c.loadTrainExamples()
            logging.info(f'欢迎游玩 {args.env}，正在启动学习过程')
        c.learn()
    elif args.mode.lower() == 'test' and args.checkpoint:
        c = Coach(g, nnet, args, player=1)
        logging.info(f'欢迎游玩 {args.env}，正在启动推理过程')
        c.infer()

if __name__ == "__main__":
    main()

# python xot_mcts/main.py --env game24 --mode train --training_env game24/data/train.csv --numMCTSSims 5000 --arenaCompare 100 --numEps 10 --numIters 3
# python xot_mcts/main.py --env game24 --mode test --test_env game24/data/test.csv  --numMCTSSims 2000 --arenaCompare 100 --multi_sol 0
