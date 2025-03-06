from unsloth import FastLanguageModel
from transformers import AutoTokenizer, BitsAndBytesConfig
from vllm import SamplingParams
import pandas as pd
from Code.Unsloth import get_model_and_tokenizer
from Code.Reward import extract_xml_answer, keep_by_replacement
from Code.Verifier import VerifierMaze
import numpy as np

SYSTEM_PROMPT = """
你是一个玩家，目前你需要通关一个迷宫游戏。这个迷宫游戏的地图是4*4的方格世界，其中你操控的智能体总是以(0,0)为起点，剩下的15个方格中，有1个方格是终点，
有2个方格是障碍物。你需要操控智能体进行移动，其中："R"使智能体的纵坐标加1；"L"使智能体的纵坐标减1；
"U"使智能体的横坐标减1；"D"使智能体的横坐标加1。需要注意的是，每一个障碍物会在它们所在的初始列进行纵向的来回
移动（每次移动的长度是1），初始移动方向是纵坐标+1，当碰到迷宫的边界会改变移动方向。为了通关游戏，你需要理解和注意：
1.操控智能体顺利走到终点。
2.不能碰到纵向来回移动的障碍物，否则游戏直接失败。
3.障碍物的移动范围是它所在的列。例如有一个初始位置为(2,1)的障碍物，它的移动范围为(0,1)到(3,1)。
现在，请你将动作序列作为答案，答案的格式严格只能含有RLUD这四个字符，如"RLUD"。你需要把把答案放在特定的格式中，即<answer>你的答案</answer>"""


# 导入模型
model, tokenizer = get_model_and_tokenizer()

test_data = pd.read_csv("env/processed_test_data.csv")

text = [tokenizer.apply_chat_template([
    {'role': "system", 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': instruct+map}
],tokenize = False, add_generation_prompt = True) for instruct, map in zip(test_data["instruct"],test_data["map"])]

sampling_params = SamplingParams(
    temperature = 0.5,
    top_p = 1.0,
    max_tokens = 2048
)

output_before_GRPO = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = None,
)

output_after_GRPO = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("Code/outputs/checkpoint-200"),
)

response_before_GRPO = [keep_by_replacement(extract_xml_answer(output_before_GRPO[i].outputs[0].text),"ULRD") for i in range(len(test_data["map"]))]
eval = np.array([VerifierMaze(m).verify(r) for m, r in zip(test_data["map"], response_before_GRPO)])
print(response_before_GRPO)
print("Before GRPO: ", eval.sum() / len(test_data["map"]))

response_after_GRPO = [keep_by_replacement(extract_xml_answer(output_after_GRPO[i].outputs[0].text),"ULRD") for i in range(len(test_data["map"]))]
eval = np.array([VerifierMaze(m).verify(r) for m, r in zip(test_data["map"], response_after_GRPO)])
print(response_after_GRPO)
print("After GRPO: ", eval.sum() / len(test_data["map"]))
