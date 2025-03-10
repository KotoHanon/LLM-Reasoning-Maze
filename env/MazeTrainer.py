import numpy as np
import pandas as pd
from Maze import Maze

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        actor_str = ""
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        action_list = ["up", "down", "right", "left"]
        return action, action_list[action]

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

def train():
    for episode in range(50):
        # 初始化观测
        observation = env.reset()
        action_seq = []
        label = 0
        while True:
            # 刷新环境
            env.render()

            # 先通过强化学习算法按照既定策略选择一个动作
            action, action_str = RL.choose_action(str(observation))
            action_seq.append(action_str)

            # 根据选择的动作，调用环境返回下一个状态，奖励以及结束标志位
            observation_, reward, done, _ = env.step(action)
            if reward == 1:
                label = 1

            # 调用强化学习算法的训练过程
            RL.learn(str(observation), action, reward, str(observation_))

            # 将当前获取的下一步状态设定为新的状态，并准备下一个循环
            observation = observation_

            # 按照标志位结束while
            if done:
                break
        trajecotry_data.append({
            "action_seq": action_seq,
            "label": label
        })
    # 游戏结束
    print('game over')
    env.destroy()

if __name__ == "__main__":
    train_data = [
        {
            "map": [],
            "action_seq": [],
        }
    ]
    for round in range(100):
        trajecotry_data = [{
            "action_seq": [],
            "label": []
        }]

        tmp_train_data = [
            {
                "map": [],
                "action_seq": [],
            }
        ]


        min_len = 1000
        env = Maze()
        RL = QLearningTable(actions=list(range(env.n_actions)))
        init_state = env.get_inital_state()
        env.after(100, train)
        env.mainloop()
        # 输出trajecotry_data中label = 1的数据
        for trajecotry_data in trajecotry_data:
            if trajecotry_data["label"] == 1:
                min_len = len(trajecotry_data["action_seq"])
                min_len = min(min_len, len(trajecotry_data["action_seq"]))
                if min_len == len(trajecotry_data["action_seq"]):
                    tmp_train_data.pop()
                    tmp_train_data.append({
                        "map": init_state,
                        "action_seq": trajecotry_data["action_seq"]
                    })

        train_data.append({
            "map":tmp_train_data[0]["map"],
            "action_seq": tmp_train_data[0]["action_seq"]
        })
        # 对train_data中map和action_seq分量同时去重
        print(train_data)

    df_train = pd.DataFrame(train_data)
    df_train.to_csv("train_data.csv", index=False)
