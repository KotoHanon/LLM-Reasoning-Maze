import pandas as pd

data = pd.read_csv("../env/train_data.csv")[1:]
def map_process(s):
    s = s.replace("[","").replace("]","").replace(".","").replace(" ","").replace("-1","3").replace('"','')
    s += "\n"
    return s
def action_process(s):
    return s.replace("'down'","D").replace("'up'","U").replace("'left'","L").replace("'right'","R").replace(",","").replace("[","").replace("]","").replace(" ","")

data["map"] = data["map"].apply(map_process)
data["action_seq"] = data["action_seq"].apply(action_process)

data = data[data["action_seq"].str.len() <= 6] # 筛选难度合适的
data = data[data["map"].str.len() >= 16] # 保证地图不为空

data[:-50].to_csv("../env/train_data.csv",index=False)
data[-50:].to_csv("../env/test_data.csv",index=False)

# 读取 CSV 文件
file_path = "../env/train_data.csv" 
df = pd.read_csv(file_path)

# 定义解析地图的函数
def parse_map(map_string):
    rows = map_string.split("\n")
    positions = {"agent": None, "goal": None, "obstacles": []}

    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell == '1':
                positions["agent"] = (x, y)
            elif cell == '2':
                positions["goal"] = (x, y)
            elif cell == '3':
                positions["obstacles"].append((x, y))

    # 生成英文描述
    agent_pos = f"The agent is at {positions['agent']}." if positions["agent"] else "No agent found."
    goal_pos = f"The goal is at {positions['goal']}." if positions["goal"] else "No goal found."
    obstacle_pos = (
        f"Obstacles are located at {', '.join(map(str, positions['obstacles']))}."
        if positions["obstacles"]
        else "No obstacles present."
    )

    return f"{agent_pos} {goal_pos} {obstacle_pos}"

# 处理 "map" 列
df["instruct"] = df["map"].apply(parse_map)


# 保存处理后的 CSV 文件
output_file_path = "../env/processed_train_data.csv"
df.to_csv(output_file_path, index=False)

print(f"Processed file saved as: {output_file_path}")

file_path = "../env/test_data.csv" 
df = pd.read_csv(file_path)
df["instruct"] = df["map"].apply(parse_map)

output_file_path = "../env/processed_test_data.csv"
df.to_csv(output_file_path, index=False)

print(f"Processed file saved as: {output_file_path}")
