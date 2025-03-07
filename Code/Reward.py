from datasets import Dataset, load_dataset
from Verifier import VerifierMaze
import re


SYSTEM_PROMPT = """
你是一个玩家，目前你需要通关一个迷宫游戏。这个迷宫游戏的地图是4*4的方格世界，其中你操控的智能体总是以(0,0)为起点，剩下的15个方格中，有1个方格是终点，
有2个方格是障碍物。你需要操控智能体进行移动，其中："R"使智能体的纵坐标加1；"L"使智能体的纵坐标减1；
"U"使智能体的横坐标减1；"D"使智能体的横坐标加1。需要注意的是，每一个障碍物会在它们所在的初始列进行纵向的来回
移动（每次移动的长度是1），初始移动方向是纵坐标+1，当碰到迷宫的边界会改变移动方向。为了通关游戏，你需要理解和注意：
1.操控智能体顺利走到终点。
2.不能碰到纵向来回移动的障碍物，否则游戏直接失败。
3.障碍物的移动范围是它所在的列。例如有一个初始位置为(2,1)的障碍物，它的移动范围为(0,1)到(3,1)。
现在，请你将动作序列作为答案，答案的格式严格只能含有RLUD这四个字符，如"RLUD"。让我们一步一步来思考。请注意，你需要将推理的内容放在
<reasoning>
...
</reasoning>中，而将你的答案放在
<answer>
...
</answer>中。
"""

XML_COT_FORMAT = """\ 
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(response):
    return response.split("<answer>")[-1].split("<answer>")[0].strip()

def get_maze_map():
    data = load_dataset("csv", data_files="../env/processed_train_data.csv", split="train")
    data = data.map(
        lambda x:{
            "prompt": [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x["instruct"], 'map': x["map"]}
            ],
            "answer": x["action_seq"]
        }
    )

    return data

dataset = get_maze_map()

def correct_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    m = prompts[0][-1]['map']
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(response) for response in responses]
    print('-' * 20, f"Map:\n{m}", f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}",
          f"\nExtracted:\n{extracted_responses[0]}")
    print([VerifierMaze(m).verify(keep_by_replacement(extracted_response, "ULDR")) for extracted_response in extracted_responses])
    return [2.0 if VerifierMaze(m).verify(keep_by_replacement(extracted_response, "ULDR")) == 1 else 0.0 for extracted_response in extracted_responses]

def length_reward_func(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(response) for response in responses]
    return [0.5 if len(keep_by_replacement(extracted_response, "ULDR")) <= len(answer[0]) else 0.0 for response in responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def action_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r'^[A-Z]+$'
    responses = [completion[0]['content'] for completion in completions]
    answer = [extract_xml_answer(r) for r in responses]
    return [0.5 if re.match(pattern, keep_by_replacement(r, "ULDR")) else 0.0 for r in answer]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n<answer>\n")[-1]) * 0.001 # 控制答案的长度
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001 # 控制答案的长度
    return count

def xml_count_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    return [count_xml(r) for r in responses]

#筛动作序列
def keep_by_replacement(s, allowed_chars):
    for char in s:
        if char not in allowed_chars:
            s = s.replace(char, '')
    return s

    
    


