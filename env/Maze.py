# 导入库信息
import numpy as np
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
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _get_random_position(self):
        x = random.randint(0, MAZE_W - 1) * UNIT + 20
        y = random.randint(0, MAZE_H - 1) * UNIT + 20
        return np.array([x, y])

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

        self.hell1_center = self._get_random_position()
        while np.array_equal(self.hell1_center, origin):
            self.hell1_center = self._get_random_position()

        self.hell1 = self.canvas.create_rectangle(
            self.hell1_center[0] - 15, self.hell1_center[1] - 15,
            self.hell1_center[0] + 15, self.hell1_center[1] + 15,
            fill='black')

        self.hell2_center = self._get_random_position()
        while np.array_equal(self.hell2_center, origin) or np.array_equal(self.hell2_center, self.hell1_center):
            self.hell2_center = self._get_random_position()

        self.hell2 = self.canvas.create_rectangle(
            self.hell2_center[0] - 15, self.hell2_center[1] - 15,
            self.hell2_center[0] + 15, self.hell2_center[1] + 15,
            fill='black')

        oval_center = self._get_random_position()
        while np.array_equal(oval_center, origin) or np.array_equal(oval_center, self.hell1_center) or np.array_equal(oval_center, self.hell2_center):
            oval_center = self._get_random_position()

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

    def _move_hell_nodes(self):
        hell1_coords = self.canvas.coords(self.hell1)
        if hell1_coords[1] <= 0 or hell1_coords[3] >= MAZE_H * UNIT:
            self.hell1_direction *= -1
        self.canvas.move(self.hell1, 0, self.hell1_direction * UNIT)

        hell2_coords = self.canvas.coords(self.hell2)
        if hell2_coords[1] <= 0 or hell2_coords[3] >= MAZE_H * UNIT:
            self.hell2_direction *= -1
        self.canvas.move(self.hell2, 0, self.hell2_direction * UNIT)

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        self.canvas.coords(self.hell1, self.hell1_center[0] - 15, self.hell1_center[1] - 15,
                           self.hell1_center[0] + 15, self.hell1_center[1] + 15)
        self.canvas.coords(self.hell2, self.hell2_center[0] - 15, self.hell2_center[1] - 15,
                           self.hell2_center[0] + 15, self.hell2_center[1] + 15)
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])
        self._move_hell_nodes()
        s_ = self.canvas.coords(self.rect)

        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False
        info = None
        return s_, reward, done, info
    # 刷新当前环境
    def render(self):
        time.sleep(0.1)
        self.update()

    # 定义一个关于环境的二维数组，用来表示当前迷宫的状态。1表示智能体的位置，0表示空格，-1表示地狱，2表示天堂
    def get_inital_state(self):
        state = np.zeros((MAZE_W, MAZE_H))
        s = self.canvas.coords(self.rect)
        ov = self.canvas.coords(self.oval)
        he1 = self.canvas.coords(self.hell1)
        he2 = self.canvas.coords(self.hell2)
        state[int(s[0] / UNIT)][int(s[1] / UNIT)] = 1
        state[int(ov[0] / UNIT)][int(ov[1] / UNIT)] = 2
        state[int(he1[0] / UNIT)][int(he1[1] / UNIT)] = -1
        state[int(he2[0] / UNIT)][int(he2[1] / UNIT)] = -1
        return state


# 刷新函数
def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done, info = env.step(a)
            if done:
                break
# 自测部分
if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()

