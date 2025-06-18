

# Everything of Thoughts (XoT): Defying the Law of Penrose Triangle for Thought Generation


![Compare](/assets/compare.jpg "Compare")
[XOT](https://github.com/microsoft/Everything-of-Thoughts-XoT)在24点游戏场景下的实验复现，添加大量中文注释，便于大陆地区研究人员理解。


## 复现流程

1. 克隆仓库并安装依赖
```bash
git clone https://github.com/Zemelee/XOT
cd XOT
conda create -n xot python=3.8
conda activate xot
pip install -r requirements.txt
```

2. 生成数据训练神经网络，测试模型
```bash
# 生成数据+训练模型
python xot_mcts/main.py --env game24 --mode train --training_env game24/data/train.csv --numMCTSSims 5000 --arenaCompare 100 --numEps 10 --numIters 3
# 测试模型
python xot_mcts/main.py --env game24 --mode test --test_env game24/data/test.csv  --numMCTSSims 2000 --arenaCompare 100 --multi_sol 0
```

3. 推理
```bash
# 在xot_all_in_one/xot/controller/llm/models.py定义需要调用的模型(基于OPENAI)
python xot_all_in_one/main.py --config xot_all_in_one/config/game24/single_sol/game24_single_xot_laststep0_revised0.yaml
```

