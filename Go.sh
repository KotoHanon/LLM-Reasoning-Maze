#!/bin/bash

# 依次执行 python 文件
python env/MazeTrainer.py # 生成数据集
python DataProcess/Process.py # 对数据集进行清洗
python Code/Trainer.py # 利用GRPO进行PEFT
python Inference.py # LoRA合并，生成结果action_seq
python Eval.py # 在Maze环境中进行评估（带有GUI）
