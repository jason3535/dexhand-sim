"""第一个真任务：让 LEAP Hand 抓住一个方块，并把过程渲染成图。

跑法: ./.venv/bin/python grasp.py
产物: grasp.png (关键帧拼图) + mujoco_menagerie/leap_hand/_scene_grasp.xml
      （那个 xml 可以直接用 viewer 打开，自己拖着玩）

注意：物体的落点由 sim.rest_height() 自动算出来 —— 贴着掌面放，不插进手里。
最初那版把方块写死在 z=0.10，那个位置在手掌**内部**，一开始就是穿模的，
跑出来的「抓住了」是穿透回弹力撑住的假象。详见 RESULTS.md。
"""

import numpy as np

import sim

# 用比基准更紧一档的位形：sweep.py 的结论是闭合幅度才是抓得牢的主因
POSE = sim.BASE_POSE * 1.3


def main():
    shots_at = [0.30, 0.75, 1.20, 1.60, 2.00, 2.39]
    r = sim.run(pose=POSE, shots_at=shots_at, keep_scene=True)

    print(f"落点        x={sim.BEST_XY[0]:.3f}  y={sim.BEST_XY[1]:.3f}  "
          f"z={r['z0']*100:.1f}cm（自动贴掌面）")
    print(f"抓稳时高度   {r['z_hold']*100:.1f}cm   物体上的接触点 {r['n_con']} 个   "
          f"法向力合计 {r['f_con']:.2f}N")
    print(f"抓握强度     {sim.verdict(r)}")
    print("判定:", "✋ 真抓住了（翻过来也掉不下去）" if r["ratio"] >= 1
          else "❌ 只是躺在掌心，一翻手就掉")

    labels = [f"t={t:.2f}s" for t in shots_at] + r["frame_labels"][len(shots_at):]
    try:
        p = sim.contact_sheet(r["frames"], sim.ROOT / "grasp.png", cols=3, labels=labels)
        print(f"关键帧拼图 -> {p}")
    except ImportError:
        print("装个 pillow 就能出拼图: uv pip install pillow")

    print(f"\n想自己拖着玩:\n  ./.venv/bin/python -m mujoco.viewer "
          f"--mjcf=mujoco_menagerie/leap_hand/_scene_grasp.xml")


if __name__ == "__main__":
    main()
