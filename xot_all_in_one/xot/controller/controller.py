# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
from .utils import *

# from .solver.io_solver import IO_Solver
from xot.env import Game24
from xot.prompter import Game24Prompter
from xot.parser import Game24Parser
from .solver.xot_solver import XoT_Solver


class Controller:
    # 用于管理执行流的控制器类，涉及到LLM、操作、提示和解析
    def __init__(self, config, gpt, game:Game24, prompter:Game24Prompter, parser:Game24Parser):
        self.config = config
        self.gpt = gpt
        self.game = game
        self.prompter = prompter
        self.parser = parser


    def initial_logs(self, config):
        if config.method == 'io' or config.method == 'cot':
            file = f'logs/{config.env}/{config.gpt.backend}_{config.gpt.temperature}_{config.method}_sample{config.param.n_generate_sample}_multi{config.multi_solution}_start{config.task.task_start_index}_end{config.task.task_end_index}.json'
        elif config.method == 'tot':
            file = f'logs/{config.env}/{config.gpt.backend}_{config.gpt.temperature}_{config.method}_propose{config.param.n_generate_sample}_value{config.param.n_evaluate_sample}_greedy{config.param.n_select_sample}_start{config.task.task_start_index}_end{config.task.task_end_index}_laststep{config.param.last_step}.json'
        elif config.method == 'got':
            file = f'logs/{config.env}/{config.gpt.backend}_{config.gpt.temperature}_{config.method}_multi{config.multi_solution}_propose{config.param.n_generate_sample}_value{config.param.n_evaluate_sample}_greedy{config.param.n_select_sample}_start{config.task.task_start_index}_end{config.task.task_end_index}_laststep{config.param.last_step}.json'
        elif config.method == 'xot':
            file = f'logs/{config.env}/{config.gpt.backend}_{config.gpt.temperature}_{config.method}_multi{config.multi_solution}_revised{config.xot.revised}_reviseTimes{config.xot.revise_times}_start{config.task.task_start_index}_end{config.task.task_end_index}_laststep{config.param.last_step}.json'
        else:
            raise ValueError("invalid method")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        return file

    # def initial_solver(self, config):
    #     if config.method == 'io':
    #         return IO_Solver(config, self.gpt, self.game, self.prompter, self.parser)
    #     elif config.method == 'cot':
    #         return CoT_Solver(config, self.gpt, self.game, self.prompter, self.parser)
    #     elif config.method == 'tot':
    #         return ToT_Solver(config, self.gpt, self.game, self.prompter, self.parser)
    #     elif config.method == 'got':
    #         return GoT_Solver(config, self.gpt, self.game, self.prompter, self.parser)
    #     elif config.method == 'xot':
    #         return XoT_Solver(config, self.gpt, self.game, self.prompter, self.parser)
    #     else:
    #         raise ValueError("invalid method")

    def run(self):
        config = self.config
        logs = []
        file = self.initial_logs(config) # 创建log文件
        solver = XoT_Solver(config, self.gpt, self.game, self.prompter, self.parser) # XoT_Solver
        # 遍历任务索引
        for idx in range(config.task.task_start_index, config.task.task_end_index):
            x = self.game.getOneTestBoard(idx) # 按序获取测试数据
            print(f'[{idx+1-config.task.task_start_index}/{config.task.task_end_index-config.task.task_start_index}] Problem: {x}')
            # solve
            revised_flag = None
            # 仅针对xot方法
            ys, info, revised_flag, _, model_call_list = solver.solve(idx)
            if config.multi_solution and config.method not in ['tot', 'got']:
                infos = [self.parser.test_output_multi(x, y, revised_flag) for y in ys]
            else:
                infos = [self.parser.test_output(x, y, revised_flag) for y in ys] # [{'r': 0, 'revised': 1}]
            # 仅针对xot方法
            info.update({'idx': idx, 'ys': ys, 'infos': infos, 'model_call': model_call_list[0], 'model_call_phase1': model_call_list[1], 'model_call_phase2': model_call_list[2]})
            # log
            logs.append(info)
            with open(file, 'w') as f:
                json.dump(logs, f, indent=4)
            