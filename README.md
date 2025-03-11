## 小项目：基于动态迷宫环境利用GRPO对QWen2.5-14B-Instruct进行PEFT
![](https://img.picui.cn/free/2025/03/08/67cc2c87b8d8c.png)

项目的灵感来源：RAGEN https://github.com/ZihanWang314/ragen

**1. 仓库clone**

`git clone https://github.com/KotoHanon/LLM-Reasoning-Maze.git`

**2. 依赖库下载**

`pip install -r requirements.txt`

**3. 运行脚本**

`bash Go.sh`

**4. 一些细节**

一张L20 + Unsloth的4-bit量化

**5. 结果**

Before GRPO Successful Rate: 0.38

After GRPO-1000steps Successful Rate: **0.42**


