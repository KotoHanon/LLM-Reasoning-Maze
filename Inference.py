from unsloth import FastLanguageModel
from transformers import AutoTokenizer, BitsAndBytesConfig
from vllm import SamplingParams
import pandas as pd
from Code.Unsloth import get_model_and_tokenizer
from Code.Reward import extract_xml_answer, keep_by_replacement
from Code.Verifier import VerifierMaze
import numpy as np

SYSTEM_PROMPT = """
现在，请你将动作序列作为答案，答案的格式严格只能含有RLUD这四个字符，如"RLUD"（"R"表示向下移动使智能体的纵坐标加1；"L"表示向上移动使智能体的纵坐标减1；
"U"表示向左移动使智能体的横坐标减1；"D"表示向右移动使智能体的横坐标加1）。让我们一步一步来思考。请注意，你需要将推理的内容放在
<reasoning>
...
</reasoning>中，而将你的答案放在
<answer>
...
</answer>中。
"""


# 导入模型
model, tokenizer = get_model_and_tokenizer()

test_data = pd.read_csv("env/processed_test_data.csv")

text = [tokenizer.apply_chat_template([
    {'role': "system", 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': instruct+map}
],tokenize = False, add_generation_prompt = True) for instruct, map in zip(test_data["instruct"],test_data["map"])]

sampling_params = SamplingParams(
    temperature = 0.0,
    top_p = 0.2,
    max_tokens = 2048
)

output_before_GRPO = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = None,
)

output_after_GRPO_1000 = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("Code/outputs/checkpoint-1000"),
)

def Eval(output, tags):
    response = [keep_by_replacement(extract_xml_answer(output[i].outputs[0].text),"ULRD") for i in range(len(test_data["map"]))]
    if output == output_after_GRPO_1000:
        dataframe = pd.DataFrame(response)
        dataframe.to_csv("Action_Seq.csv")
    eval = np.array([VerifierMaze(m).verify(r) for m, r in zip(test_data["map"], response)])
    print(f"{tags}: ", eval.sum() / len(test_data["map"]))

Eval(output_before_GRPO, "Before GRPO")
Eval(output_after_GRPO_1000, "After GRPO 1000 steps")
