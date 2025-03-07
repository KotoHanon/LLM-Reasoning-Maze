# 导入库信息
import numpy as np
import pandas as pd
import time
import sys
import random

# 设定环境信息
UNIT = 40   # 设定是像素大小为40
MAZE_H = 4  # 设置纵轴的格子数量
MAZE_W = 4  # 设置横轴的格子数量

# 创建一个迷宫类
class VerifierMaze(object):
    def __init__(self, maze_str):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.maze_str = maze_str

        # 存储各元素的位置坐标（左上x, 左上y, 右下x, 右下y格式）
        self.rect_pos = None
        self.oval_pos = None
        self.hells_pos = []

        # 存储移动方向
        self.hells_direction = []

        # 初始化迷宫布局
        self._build_maze()

    def _get_position(self, maze_str):
        position_array = self.str2array(maze_str)
        position_array = position_array.T
        rect_x, rect_y = np.where(position_array == 1)
        oval_x, oval_y = np.where(position_array == 2)
        hell_x, hell_y = np.where(position_array == 3)
        return rect_x, rect_y, oval_x, oval_y, hell_x, hell_y

    def _build_maze(self):
        # 计算各元素的初始位置（使用与原始代码相同的坐标计算逻辑）
        rect_x, rect_y, oval_x, oval_y, hell_x, hell_y = self._get_position(self.maze_str)

        # 计算矩形（玩家）初始位置
        self.rect_pos = np.array([
            rect_x[0] * UNIT + 5,  # 左上x（原代码用20-15=5）
            rect_y[0] * UNIT + 5,  # 左上y
            rect_x[0] * UNIT + 35,  # 右下x（原代码用20+15=35）
            rect_y[0] * UNIT + 35  # 右下y
        ])

        # 计算终点位置
        self.oval_pos = np.array([
            oval_x[0] * UNIT + 5,
            oval_y[0] * UNIT + 5,
            oval_x[0] * UNIT + 35,
            oval_y[0] * UNIT + 35
        ])

        # 计算陷阱位置
        for i in range(len(hell_x)):
            pos = np.array([
                hell_x[i] * UNIT + 5,
                hell_y[i] * UNIT + 5,
                hell_x[i] * UNIT + 35,
                hell_y[i] * UNIT + 35
            ])
            self.hells_pos.append(pos)

        # 初始化陷阱移动方向（与原始代码相同）
        self.hells_direction = [1, 1]  # 假设固定两个陷阱

    def str2array(self, maze_str):
        # 保持原方法不变
        array = np.zeros([4, 4])
        maze_str = maze_str.replace("\n", "").replace("\r", "")
        for i in range(4):
            for j in range(4):
                cur = 4 * j + i
                if maze_str[cur] == '1':
                    array[i][j] = 1
                elif maze_str[cur] == '2':
                    array[i][j] = 2
                elif maze_str[cur] == '3':
                    array[i][j] = 3
        return array

    def _move_hell_nodes(self):
        # 更新陷阱位置（垂直移动）
        for i in range(len(self.hells_pos)):
            # 获取当前坐标
            top_y = self.hells_pos[i][1]
            bottom_y = self.hells_pos[i][3]

            # 边界检测（与原始代码相同逻辑）
            if top_y <= 0 or bottom_y >= MAZE_H * UNIT:
                self.hells_direction[i] *= -1

            # 更新Y坐标
            delta = self.hells_direction[i] * UNIT
            self.hells_pos[i][1] += delta
            self.hells_pos[i][3] += delta

    def verify(self, action_seq: str):
        current_pos = self.rect_pos.copy()
        for action in action_seq.upper():
            # 处理移动（保持原边界检测逻辑）
            new_pos = current_pos.copy()

            if action == "U":
                if current_pos[1] > UNIT:  # 原条件 s[1] > UNIT
                    new_pos[1] -= UNIT
                    new_pos[3] -= UNIT
            elif action == "D":
                if current_pos[3] < (MAZE_H - 1) * UNIT:  # 原条件 s[1] < (MAZE_H-1)*UNIT
                    new_pos[1] += UNIT
                    new_pos[3] += UNIT
            elif action == "R":
                if current_pos[2] < (MAZE_W - 1) * UNIT:  # 原条件 s[0] < (MAZE_W-1)*UNIT
                    new_pos[0] += UNIT
                    new_pos[2] += UNIT
            elif action == "L":
                if current_pos[0] > UNIT:  # 原条件 s[0] > UNIT
                    new_pos[0] -= UNIT
                    new_pos[2] -= UNIT

            # 更新位置
            current_pos = new_pos

            # 移动陷阱（每次移动都更新）
            self._move_hell_nodes()

            # 碰撞检测（与原始代码相同顺序）
            # 检测陷阱碰撞
            hell_collision = False
            for hell in self.hells_pos:
                if np.array_equal(current_pos, hell):
                    hell_collision = True
                    break

            if hell_collision:
                return 0  # 提前终止

        # 最终判断
        if np.array_equal(current_pos, self.oval_pos):
            return 1
        else:
            return 0

# 自测部分
if __name__ == '__main__':
    data = pd.read_csv("../env/train_data.csv")
    env = VerifierMaze(data["map"][1])
    print(data["action_seq"][1])
    print(env.verify(data["action_seq"][1]))
