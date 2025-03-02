# 导入库信息
import numpy as np
import pandas as pd
import time
import sys
import random
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

# 设定环境信息
UNIT = 40   # 设定是像素大小为40
MAZE_H = 4  # 设置纵轴的格子数量
MAZE_W = 4  # 设置横轴的格子数量

# 创建一个迷宫类
class VerifierMaze(tk.Tk, object):
    def __init__(self, str):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.str = str

    def _get_position(self, str):
        position_array = self.str2array(str)
        rect_x, rect_y = np.where(position_array == 1)
        oval_x, oval_y = np.where(position_array == 2)
        hell_x, hell_y = np.where(position_array == 3)
        return rect_x, rect_y, oval_x, oval_y, hell_x, hell_y

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([20, 20])

        rect_x, rect_y, oval_x, oval_y, hell_x, hell_y = self._get_position(self.str)
        self.rect_center = np.array(rect_x[0], rect_y[0])
        self.oval_center = np.array(oval_x[0], oval_y[0])
        self.hell1_center = np.array(hell_x[0], hell_y[0])
        self.hell2_center = np.array(hell_x[1], hell_y[1])

        self.hell1 = self.canvas.create_rectangle(
            self.hell1_center[0] - 15, self.hell1_center[1] - 15,
            self.hell1_center[0] + 15, self.hell1_center[1] + 15,
            fill='black')

        self.hell2 = self.canvas.create_rectangle(
            self.hell2_center[0] - 15, self.hell2_center[1] - 15,
            self.hell2_center[0] + 15, self.hell2_center[1] + 15,
            fill='black')

        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        self.canvas.pack()

        self.hell1_direction = 1
        self.hell2_direction = 1

    def str2array(str):
        array = np.zeros([4, 4])
        str = str.replace("\n", "")
        for i in range(4):
            for j in range(4):
                cur = 4 * j + i
                if (str[cur] == '1'):
                    array[i][j] = 1
                elif (str[cur] == '2'):
                    array[i][j] = 2
                elif (str[cur] == '3'):
                    array[i][j] = 3

        return array

    def _move_hell_nodes(self):
        hell1_coords = self.canvas.coords(self.hell1)
        if hell1_coords[1] <= 0 or hell1_coords[3] >= MAZE_H * UNIT:
            self.hell1_direction *= -1
        self.canvas.move(self.hell1, 0, self.hell1_direction * UNIT)

        hell2_coords = self.canvas.coords(self.hell2)
        if hell2_coords[1] <= 0 or hell2_coords[3] >= MAZE_H * UNIT:
            self.hell2_direction *= -1
        self.canvas.move(self.hell2, 0, self.hell2_direction * UNIT)

    def verify(self, action_seq:str):
        s = self.canvas.coords(self.rect)
        s_ = None
        base_action = np.array([0, 0])
        for i in range(len(action_seq)):
            if action_seq[i] == "D":
                if s[1] > UNIT:
                    base_action[1] -= UNIT
            elif action_seq[i] == "U":
                if s[1] < (MAZE_H - 1) * UNIT:
                    base_action[1] += UNIT
            elif action_seq[i] == "R":
                if s[0] < (MAZE_W - 1) * UNIT:
                    base_action[0] += UNIT
            elif action_seq[i] == "L":
                if s[0] > UNIT:
                    base_action[0] -= UNIT

            self.canvas.move(self.rect, base_action[0], base_action[1])
            self._move_hell_nodes()
            s_ = self.canvas.coords(self.rect)
            s = s_

        if s_ == self.canvas.coords(self.oval):
            reward = 1
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
        else:
            reward = 0
        return reward

# 自测部分
if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()


