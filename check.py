"""跑通验证：加载 LEAP Hand，打印结构，空跑一段物理。"""
import pathlib
import mujoco
import numpy as np

XML = pathlib.Path(__file__).parent / "mujoco_menagerie/leap_hand/scene_right.xml"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

print(f"MuJoCo {mujoco.__version__}")
print(f"模型      : {XML.name}")
print(f"自由度 nq : {model.nq}  (广义坐标)")
print(f"速度 nv   : {model.nv}")
print(f"驱动器 nu : {model.nu}  (可控关节 = 你实际能下指令的数量)")
print(f"刚体 nbody: {model.nbody}   几何体 ngeom: {model.ngeom}")
print(f"时间步 dt : {model.opt.timestep}s")

print("\n--- 驱动器（关节）清单 ---")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    lo, hi = model.actuator_ctrlrange[i]
    print(f"  [{i:2d}] {name:<12s} 控制范围 {lo:+.3f} ~ {hi:+.3f} rad")

# 空跑 2 秒物理
n = int(2.0 / model.opt.timestep)
import time
t0 = time.perf_counter()
for _ in range(n):
    mujoco.mj_step(model, data)
dt = time.perf_counter() - t0
print(f"\n空跑 {n} 步（2 仿真秒）耗时 {dt:.2f}s → 实时倍率 {2.0/dt:.0f}x")
print(f"末态 qpos 前 8 位: {np.round(data.qpos[:8], 4)}")
print("\n✅ 跑通了。")
