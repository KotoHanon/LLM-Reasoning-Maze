from datasets import Dataset, load_dataset
import Verifier.VerifierMaze as VerifierMaze

SYSTEM_PROMPT = """
Now you are a player in a maze game where the user provides a 4x4 grid formatted as "xxxx\nxxxx\nxxxx\nxxxx\n", 
with "1" representing your controllable agent, "2" as the goal, and "3" as vertically oscillating obstacles. 
The agent (1) must navigate using "U/D/L/R" commands to reach the goal (2) without colliding with obstacles (3), 
which move downward initially and reverse direction upon hitting grid boundaries. Each agent movement triggers 
synchronized obstacle shifts within their columns, and you must return the optimal directional sequence to 
achieve success while avoiding collisions. Respond in the following format:
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
    return response.split("<answer>")[-1].split("<answer>").strip()

def get_maze_map():
    data = load_dataset(path="../env/train_data.csv")
    data = data.map(
        lambda x:{
            "prompt": [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x["map"]}
            ],
            "answer": x["action_seq"]
        }
    )

    return data

dataset = get_maze_map()

def correct_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(response) for response in responses]
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}",
          f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if VerifierMaze(q).verify(response) else 0.0 for response in responses]

def length_reward_func(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    return [0.5 if len(response) <= len(answer[0]) else 0.0 for response in responses]

    
    


