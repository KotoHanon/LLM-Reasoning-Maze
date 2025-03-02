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

data[:-10].to_csv("../env/train_data.csv",index=False)
data[-10:].to_csv("../env/test_data.csv",index=False)
