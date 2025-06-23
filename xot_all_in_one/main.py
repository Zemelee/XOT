from argparse import ArgumentParser
from functools import partial

from utils import Config, load_config
from xot import env
from xot.controller import chatgpt
from xot.controller import Controller
from xot.prompter import Game24Prompter
from xot.parser import Game24Parser
 
  
parser = ArgumentParser("Everything of Thoughts! 🎉")  
parser.add_argument('--config', type=str, required=False, help='Path to YAML configuration file.')  
args = parser.parse_args()

yaml_file = "xot_all_in_one/config/game24/single_sol/game24_single_xot_laststep1_revised1.yaml"
config = Config(load_config(yaml_file))

gpt = partial(chatgpt, temperature=config.gpt.temperature)    
game = env.Game24(test_dir=config.task.data)
prompter = Game24Prompter(last_step=config.param.last_step) # 0/1
parser = Game24Parser()
# Create the Controller
ctrl = Controller(config, gpt, game, prompter, parser)
# Run the Controller and generate the output
ctrl.run()

# python xot_all_in_one/main.py --config xot_all_in_one/config/game24/single_sol/game24_single_xot_laststep0_revised0.yaml