from datasets import Dataset, load_dataset
from Verifier import VerifierMaze
import re


SYSTEM_PROMPT = """
Now you are a player in a maze game where the user provides a 4x4 grid with the coordinate system starting at the top left corner.
The agent (1) must navigate using the "U/D/L/R" instruction to reach the goal 
(" U "means the agent is walking up, minus 1 in the ordinate;" D" means that the agent is walking down, 
and the ordinate is plus 1;" "L" means that the agent walks to the left, minus 1 on the horizontal coordinate; "
R" means that the agent walks to the right, plus 1) (2) on the horizontal coordinate without colliding 
with the obstacle (3), which move downward initially and reverse direction upon hitting grid boundaries. 
Each agent movement triggers synchronized obstacle shifts within their columns, and you must return the 
optimal directional sequence to achieve success while avoiding collisions. Your answer should follow the 
strict format as "UDLR", Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
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
    print([VerifierMaze(m).verify(extracted_response) for extracted_response in extracted_responses])
    return [2.0 if VerifierMaze(m).verify(extracted_response) == 1 else 0.0 for extracted_response in extracted_responses]

def length_reward_func(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    return [0.5 if len(response) <= len(answer[0]) else 0.0 for response in responses]

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
    return [0.5 if re.match(pattern, r) else 0.0 for r in answer]

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

    
    


