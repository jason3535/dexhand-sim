# dexhand-sim · 灵巧手仿真起步

在买任何硬件之前，先用仿真把想法过一遍。环境已装好，M4 本地 CPU 跑得动。

## 跑起来

刚 clone 下来的话先建环境（仓库里不含 `.venv/` 和 `mujoco_menagerie/`，一共 135M）：

```bash
./setup.sh    # 稀疏拉三只手的模型 + 建 venv 装 MuJoCo，几分钟
```

然后：

```bash
cd ~/CascadeProjects/dexhand-sim

./.venv/bin/python check.py       # 环境自检 + 打印 LEAP Hand 的 16 个关节
./.venv/bin/python grasp.py       # 抓一个方块，出 grasp.png
./.venv/bin/python calibrate.py   # 掌心哪个落点抓得最牢（7×7 热力图，~5s）
./.venv/bin/python sweep.py       # 形状/摩擦/位形/单关节 四组扫描（~55s）
./.venv/bin/python gestures.py    # 18 个手势：拼图 + 31s 动画 + viewer 关键帧（~50s）
./.venv/bin/python gestures.py --sheet   # 只出拼图，调姿势时用（~8s）
```

交互式查看器（鼠标拖视角，双击选中物体后 Ctrl+右键拖可以推它）：

```bash
./.venv/bin/python -m mujoco.viewer --mjcf=mujoco_menagerie/leap_hand/scene_right.xml
# 想看带方块的场景（grasp.py 跑完才会生成）：
./.venv/bin/python -m mujoco.viewer --mjcf=mujoco_menagerie/leap_hand/_scene_grasp.xml
# 想逐个切手势看（gestures.py 跑完才会生成，左面板 Keyframe 里 18 个）：
./.venv/bin/python -m mujoco.viewer --mjcf=mujoco_menagerie/leap_hand/_scene_gestures.xml
```

viewer 常用键：`空格` 暂停 / `→` 单步 / `Backspace` 复位 / `Tab`、`Shift+Tab` 开关左右面板 /
`C` 显示接触点 / `F` 显示接触力 / `T` 半透明 / `[` `]` 切相机 / `F1` 全部快捷键。
MacBook 触控板：单指拖=转视角，双指点住拖=平移，双指上下滑=缩放；
双击选中物体后 `Ctrl`+双指拖 = 推它（等价于外接鼠标的 Ctrl+右键拖）。

> 别用 `.venv/bin/mjpython`，这个 venv 是 uv 装的、没有 framework 版 libpython，
> mjpython 起不来（`Library not loaded: @rpath/libpython3.12.dylib`）。
> 上面的 `python -m mujoco.viewer` 在本机是好使的。

## 现在有什么

- `setup.sh` — 从零重建环境（下面两个都不在仓库里）
- `.venv/` — Python 3.12 + MuJoCo 3.11
- `mujoco_menagerie/` — 官方模型库（稀疏 checkout，只拉了三只手，固定在 `c1a4eeb`）
  脚本生成的 `_scene_*.xml` 也落在这里面 —— 因为 mesh 是相对路径，场景 XML 必须和模型同目录
  - `leap_hand/` — **LEAP Hand，16 DoF**，就是硬件上最推荐先买的那只
  - `shadow_hand/` — Shadow Hand，24 DoF，工业标杆（真机 $100k+）
  - `wonik_allegro/` — Allegro Hand，16 DoF（真机 $15k+）
- `check.py` — 环境自检
- `sim.py` — **公共仿真层**：搭场景、把物体正确放到掌心、跑抓取 + 拽拉测试
- `grasp.py` — 单次抓取演示，出 `grasp.png`
- `calibrate.py` — 扫掌心落点，出 `calibrate.csv` + 热力图
- `sweep.py` — 四组批量实验，出 `results.csv` / `sweep_shapes.png` / `sweep_output.txt`
- `views.py` — 出 `hand_views.png`：这只手张开/闭合的三视图，看坐标系用
- `gestures.py` — 18 个手势（握拳/点赞/剪刀/OK/逐指对指/波浪…），
  出 `gestures.png` + `gestures.mp4` + viewer 用的关键帧场景。
  「捏」不是手写角度，是用运动学求解拇指去够每根手指的指尖
- **`RESULTS.md` — 第 1 步的实验结论，先读这个**

## 坐标系（免得再踩坑）

掌面朝 **+z**，高度 **z≈0.129–0.135**；四指从 x≈0 伸向 +x 再往上卷；拇指在 −y 侧横过来。
物体必须放在掌面**上方**——`sim.rest_height()` 会自动二分算出贴掌面的高度。

## 已验证

```
自由度 16 / 驱动器 16 / 时间步 2ms，CPU 实时倍率 ~120x（空载）
抓握强度评分口径：抓稳后沿掌面法向加力拽，记录拽走时的力 ÷ 物体自重
  基准位形 ×1.3、方块 4.4cm  →  27.0N（55×自重），翻手也掉不下去
```

⚠️ 最初 README 里写的「28 个接触点，方块未落地 ✋」是**错的**：
那版把方块放在了手掌内部，一开始就穿模 6mm，是穿透回弹力顶住的。
详细纠错和证据见 `RESULTS.md` 第 0 节。

## 第 1 步的结论（详见 RESULTS.md）

- **抓握 = 几何包络 + 法向力，摩擦是二阶项**。往上拔的方向摩擦完全没用（μ 扫 60 倍，强度不变）；
  只有沿掌面横推才吃摩擦。
- **闭合幅度是主变量**，×1.0→×1.8 抓握强度从 4× 涨到 116×，中间有个"从贴着变成包过去"的阈值。
- **落点比什么都敏感**，偏 2cm 就从 40× 掉到 4×。
- **接触点数量和抓得牢基本无关**（r=+0.25），法向力才相关（r=+0.91）。
- 16 个关节里这次抓取真正出力的只有 6 个，无名指全程没碰到物体。
- ⚠️ MuJoCo 坑：接触对摩擦取两个 geom 的**最大值**。只调物体的 friction，
  低于手的 0.2 / 指尖 0.5 时完全没效果。
- **开环下发一组角度，手到不了那个位置**：残余 ~6°，握拳时 14°。
  而且 kp 从 3 加到 120 这个误差纹丝不动 —— 是手指**自碰撞**顶住的，不是伺服软。
  得用迭代补偿（`cmd += 0.6 × 误差`）把指令抬上去，增益取 1.0 会来回震荡。

## 接下来三步

**第 1 步 · 改参数，建立直觉** — ✅ 已完成，见 `RESULTS.md`。
还想继续挖的话：加**扭矩方向**的拽拉测试（真实失败模式多是"转出去"），
以及把手的 geom friction 一起改，做一次真正的"湿肥皂"实验。

**第 2 步 · 换任务：手内旋转（1–2 周）**
抓取是入门题，真正区分灵巧手和夹爪的是 **in-hand reorientation**（不借助外力，
只靠手指把物体在掌心转到指定朝向）。这是 OpenAI 魔方、LEAP Hand 论文的核心任务。
先用脚本做不到——这就是为什么需要 RL。

> 带着第 1 步的结论去做第 2 步会顺很多：既然系统对毫米级差异是混沌的（同条件 5 次
> 微扰动，结果 2.9×–39.8×），那 reward 的方差主要来自物理本身，不是算法。
> 实验设计上**从一开始就要多种子 + 报区间**。

**第 3 步 · 上 RL**
装 `mujoco_playground`（DeepMind 出的，自带 LeapCubeReorient 环境）或 ManiSkill。
> ⚠️ **这一步 Mac 会卡住**：MJX/JAX 的大规模并行需要 CUDA，M4 上只能跑 CPU 版，
> 慢几十倍。到这一步再租一台 4090 云主机（几块钱一小时），不用现在买。

## 什么时候该买硬件

跑完第 2 步，你应该能回答这三个问题。答不上来就别买：

1. 你要解决的问题，**夹爪能不能做**？（能的话就别碰灵巧手）
2. 你需要几个自由度？触觉是必需还是加分？
3. 你的策略是**遥操作示教**还是 **RL**？前者要配数采方案，后者要配 GPU。

答得上来了，第一只手买 [LEAP Hand v2](https://v2.leaphand.com/)（$200）。
