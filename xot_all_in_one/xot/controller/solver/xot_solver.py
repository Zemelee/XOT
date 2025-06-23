# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import numpy as np
from collections import Counter
from .MCTS import MCTS
from .Coach import Coach
from .pytorch_game24.NNet import NNetWrapper as nn
from xot.env import Game24
from xot.prompter import Game24Prompter
from xot.parser import Game24Parser

class XoT_Solver:
    """
    XoT_Solver类旨在使用蒙特卡洛树搜索(MCTS)、神经网络(NN)和教练机制的组合来解决各种游戏。
    支持单一解和多解，并能根据反馈修正解决方案。
    属性:
        args: 包含各种参数的配置对象。
        gpt: 用于生成提示的GPT模型实例。
        game: 要解决的游戏实例。
        prompter: 用于生成提示的类实例。
        parser: 用于解析动作和思路的类实例。
        nmcts: MCTS实例。
        c: 教练(Coach)实例。
        to_print: 布尔值，指示是否打印调试信息。
    """
    def __init__(self, args, gpt, game:Game24, prompter:Game24Prompter, parser:Game24Parser, to_print=False):
        # 使用给定的参数、GPT模型、游戏、提示器、解析器和打印选项初始化XoT_Solver。
        self.args = args
        self.gpt = gpt
        self.game = game
        self.prompter = prompter
        self.parser = parser
        self.nmcts, self.c = self.initial_xot(args)
        self.to_print = to_print
    
    def initial_xot(self, args):
        # 根据参数中指定的游戏环境初始化神经网络和MCTS。        
        nnet = nn(self.game)
        # 加载博弈数据训练的模型
        nnet.load_checkpoint(folder=self.args.model.checkpoint, filename=self.args.model.filename)
        nmcts = MCTS(self.game, nnet, args)
        c = Coach(self.game, nnet, args)
        return nmcts, c
    
    def multi_solve_before_revision(self, x):
        # 在进行任何修正之前，为游戏求解多个解。
        nmcts_modelcall_before = self.nmcts.getModelCall()
        player = lambda x: np.argmax(self.nmcts.getActionProb(x, temp=0, step=0))
        problem_state, getGameEnded, actions_idx, actions = self.c.generate_thoughts(x, player)
        actions_list, actions_candicates_list = [], []
        for i in range(self.args.xot.multi_solution_exploration):
            selected_ac_seq, _ = self.nmcts.inferSinglePlayer(problem_state, step=0, seed=i)
            if selected_ac_seq is not None:
                actions_candicates_list.append(str(selected_ac_seq))
        
        count = Counter(actions_candicates_list)  
        actions_list = [ast.literal_eval(item) for item, _ in count.most_common(3)] 
        nmcts_modelcall_after = self.nmcts.getModelCall()
        model_call_phase1 = nmcts_modelcall_after - nmcts_modelcall_before
        thoughts_list = []
        for actions in actions_list:
            try:
                thoughts_list.append(self.parser.action_to_thoughs(actions, x))
            except:
                continue
        if self.to_print:
            print('xot_solve -- 思路: ', thoughts_list)
        prompt, _ = self.prompter.xot_prompt_multi_wrap(x, thoughts_list)
        instruct = self.prompter.get_instruction_prompt()
        samples = self.gpt(prompt, instruct, n=self.args.param.n_generate_sample, stop=self.args.gpt.stop)
        return samples, thoughts_list, actions, model_call_phase1
    

    def single_solve_before_revision(self, x):
        # 在进行修正之前，为游戏求解单个解。
        player = lambda x: np.argmax(self.nmcts.getActionProb(x, temp=0, step=0))
        nmcts_modelcall_before = self.nmcts.getModelCall()
        problem_state, getGameEnded, actions_idx, actions = self.c.generate_thoughts(x, player)
        nmcts_modelcall_after = self.nmcts.getModelCall()
        model_call_phase1 = nmcts_modelcall_after - nmcts_modelcall_before
        thoughts = self.parser.action_to_thoughs(actions, x) # 包含left表达式的列表

        # if self.to_print:
        print('xot_solve -- 思路: ', thoughts)
        prompt, _ = self.prompter.xot_prompt_wrap(x, thoughts) # 将思路包裹在24点游戏场景的提示词中，让LLM根据思路给出

        instruct = self.prompter.get_instruction_prompt() # Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
        samples = self.gpt(prompt, instruct, n=self.args.param.n_generate_sample, stop=self.args.gpt.stop)
  
        return samples, thoughts, actions, model_call_phase1


    def solve_single_revision(self, x, samples, thoughts, actions, model_call_phase1, model_call_phase2):
        # 基于反馈修正单个解。
        instruct = self.prompter.get_instruction_prompt()
        for revise_count in range(self.args.xot.revise_times):
            infos = [self.parser.test_output(x, y, None) for y in samples]
            isCorrect = infos[0]['r']
            # 终止条件
            if isCorrect:
                model_call = model_call_phase1 + model_call_phase2
                if revise_count == 0:
                    return samples, {}, False, None, [model_call, model_call_phase1, model_call_phase2]
                else:
                    return samples, {}, revise_count, revised_state, [model_call, model_call_phase1, model_call_phase2]

            if not isCorrect:
                revised_prompt, _ = self.prompter.xot_prompt_revised_wrap(x, thoughts)
                revised_samples = self.gpt(revised_prompt, instruct, n=self.args.param.n_generate_sample, stop=self.args.gpt.stop)
                
                if self.to_print:
                    print('revised_prompt', revised_prompt)
                    print('revised_samples', revised_samples)
                if 'wrong' in revised_samples[0].lower() or 'incorrect' in revised_samples[0].lower():
                    try:
                        if 'all steps are wrong' in revised_samples[0].lower():
                            incorrect_step = 1
                            if self.to_print:
                                print('all the steps are wrong')
                        else:
                            incorrect_step = int(revised_samples[0].split('is wrong')[0].strip().split(']')[0][-1])
                        revised_state = self.parser.get_revised_state(x, thoughts, incorrect_step)
                        if self.to_print:
                            print('incorrect_step', incorrect_step)
                            print('revised_state', revised_state)
                        if self.args.env == 'game24':
                            if incorrect_step > 1:
                                ac_seq = actions_idx[:incorrect_step-1]
                                state = x
                                for i in range(len(ac_seq)):
                                    state, _ = self.game.getNextState(state, actions_idx[i])
                                revised_state = state

                        nmcts_modelcall_before = self.nmcts.getModelCall()
                        player = lambda x: np.argmax(self.nmcts.getActionProb(x, temp=0, step=0))

                        problem_state, getGameEnded, actions_idx, actions_revised = self.c.generate_thoughts(revised_state, player)
                        nmcts_modelcall_after = self.nmcts.getModelCall()
                        model_call_phase2 += nmcts_modelcall_after - nmcts_modelcall_before

                        actions_after_revised = actions[:incorrect_step-1]
                        actions_after_revised.extend(actions_revised)
                        
                        thoughts_revised = self.parser.action_to_thoughs(actions_after_revised, x)
            
                        if self.to_print:
                            print('actions_revised', actions_revised)
                            print('actions_after_revised', actions_after_revised)
                            print('thoughts_revised', thoughts_revised)
                            print('xot_solve -- revised thoughts: ', thoughts_revised)

                        prompt, _ = self.prompter.xot_prompt_wrap(x, thoughts_revised)
                        instruct = self.prompter.get_instruction_prompt()
                        samples = self.gpt(prompt, instruct, n=self.args.param.n_generate_sample, stop=self.args.gpt.stop)
                    except:
                        continue
        # after N revised times
        model_call = model_call_phase1 + model_call_phase2
        return samples, {}, revise_count+1, None, [model_call, model_call_phase1, model_call_phase2]  


    def solve_multi_revision(self, x, samples, thoughts_list, actions, model_call_phase1, model_call_phase2_total):
        # 根据反馈修改多个解决方案
        instruct = self.prompter.get_instruction_prompt()
        infos = [self.parser.test_output_multi(x, y, [None, None, None]) for y in samples]
        isCorrect_list = infos[0]['r']
        
        if self.to_print:
            print('x', x)
            print('infos', infos)
            print('thoughts_list', thoughts_list)
            print('samples', samples)
            print('isCorrect_list', isCorrect_list)

        thoughts_revised_list = thoughts_list[:]
        revise_flags = [False] * len(isCorrect_list)

        for idx, r_ in enumerate(isCorrect_list):
            if idx >= len(thoughts_list): # gpt提供的ans可能多于给定的思想
                break
            isCorrect = isCorrect_list[r_]['r'] 
            if self.to_print:
                print('isCorrect', isCorrect)
    
            if not isCorrect:
                revise_flags[idx] = True
                thoughts = thoughts_list[idx]
                
                revised_prompt, _ = self.prompter.xot_prompt_revised_wrap(x, thoughts)
                revised_samples = self.gpt(revised_prompt, instruct, n=self.args.param.n_generate_sample, stop=self.args.gpt.stop)
                
                if self.to_print:
                    print('thoughts', thoughts)
                    print('revised_prompt', revised_prompt)
                    print('revised_samples', revised_samples)
        
                if 'wrong' in revised_samples[0].lower()or 'incorrect' in revised_samples[0].lower():
                    try:
                        if 'all steps are wrong' in revised_samples[0].lower():
                            incorrect_step = 1
                            if self.to_print:
                                print('all steps are wrong')
                        else:
                            incorrect_step = int(revised_samples[0].split('is wrong')[0].strip().split(']')[0][-1])
                        
                        revised_state = self.parser.get_revised_state(x, thoughts, incorrect_step)
                        if self.to_print:
                            print('incorrect_step', incorrect_step)
                            print('revised_state', revised_state)

                        if self.args.env == 'game24':
                            if incorrect_step > 1:
                                ac_seq = actions_idx[:incorrect_step-1]
                                state = x
                                for i in range(len(ac_seq)):
                                    state, _ = self.game.getNextState(state, actions_idx[i])
                                revised_state = state
              
                        nmcts_modelcall_before = self.nmcts.getModelCall()
                        player = lambda x: np.argmax(self.nmcts.getActionProb(x, temp=0, step=0))
                        problem_state, getGameEnded, actions_idx, actions_revised = self.c.generate_thoughts(revised_state, player)
                        nmcts_modelcall_after = self.nmcts.getModelCall()
                        model_call_phase2_total += nmcts_modelcall_after - nmcts_modelcall_before
                        
                        actions_after_revised = actions[:incorrect_step-1]
                        actions_after_revised.extend(actions_revised)
                        
                        thoughts_revised = self.parser.action_to_thoughs(actions_after_revised, x)
                        thoughts_revised_list[idx] = thoughts_revised

                        if self.to_print:
                            print('actions_revised', actions_revised)
                            print('actions_after_revised', actions_after_revised)
                            print('thoughts_revised', thoughts_revised)
                    
                    except:
                        model_call_phase2_total += 0
                else:
                    model_call_phase2_total += 0

        prompt, _ = self.prompter.xot_prompt_multi_wrap(x, thoughts_revised_list)
        instruct = self.prompter.get_instruction_prompt()
        samples_revised = self.gpt(prompt, instruct, n=self.args.param.n_generate_sample, stop=self.args.gpt.stop)
        model_call = model_call_phase1 + model_call_phase2_total
        return samples_revised, {}, revise_flags, None, [model_call, model_call_phase1, model_call_phase2_total]
    

    def solve(self, idx):
        # 解决游戏的主要方法。它首先生成解决方案，然后根据需要进行修改
        x = self.game.getOneTestBoard(idx) # 获取输入
        self.nmcts.reset()
        model_call_phase1, model_call_phase2 = 0, 0
        self.c.game.total_game_step = self.args.task.total_game_step
        
        if self.args.multi_solution:
            self.nmcts.args.numMCTSSims = self.args.xot.multi_numMCTSSims
        else:
            self.nmcts.args.numMCTSSims = self.args.xot.numMCTSSims

        if not self.args.multi_solution:
            samples, thoughts, actions, model_call_phase1 = self.single_solve_before_revision(x)
        else:
            samples, thoughts_list, actions, model_call_phase1 = self.multi_solve_before_revision(x)
        
        if not self.args.xot.revised:
            model_call = model_call_phase1 + model_call_phase2
            return samples, {}, False, None, [model_call, model_call_phase1, model_call_phase2]
        else:  
            # Update Config For Revision
            self.c.game.total_game_step = self.args.xot.revise_total_game_step
            self.nmcts.args.numMCTSSims = self.args.xot.revise_numMCTSSims
            if self.args.xot.revised and not self.args.multi_solution:
                return self.solve_single_revision(x, samples, thoughts, actions, model_call_phase1, model_call_phase2)
                
            if self.args.xot.revised and self.args.multi_solution:
                return self.solve_multi_revision(x, samples, thoughts_list, actions, model_call_phase1, model_call_phase2)
            