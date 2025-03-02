import pandas as pd

data = pd.read_csv("../env/train_data.csv")[1:]
def str_process(str):
    return str.replace("[","").replace("]","").replace(".","")
data["map"] = data["map"].apply(str_process)

data.to_csv("../env/train_data.csv",index=False)