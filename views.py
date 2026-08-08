"""画出这只手张开/闭合的三视图，用来搞清楚坐标系朝哪。

跑法: ./.venv/bin/python views.py
产物: hand_views.png  (上排=张开, 下排=闭合; 左=俯视+Z 中=正视+X 右=侧视-Y)

看完你会知道：掌面朝 +z、四指从 x≈0 伸向 +x 再往上卷、拇指在 -y 侧。
"""

import mujoco
import numpy as np

import sim

XML = """
<mujoco model="views">
  <include file="right_hand.xml"/>
  <statistic center="0 0 0.10" extent="0.30"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".25 .3 .35" rgb2="0 0 0" width="32" height="512"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 0.6" dir="-0.4 0.4 -1" diffuse="1 1 1"/>
    <light pos="-0.3 0.3 0.6" dir="0.4 -0.4 -1" diffuse=".6 .6 .6"/>
    <camera name="topZ"   pos="0 0 0.55"     xyaxes="1 0 0 0 1 0"/>
    <camera name="frontX" pos="0.55 0 0.10"  xyaxes="0 1 0 0 0 1"/>
    <camera name="sideY"  pos="0 -0.55 0.10" xyaxes="1 0 0 0 0 1"/>
  </worldbody>
</mujoco>
"""


def main():
    path = sim.HAND_DIR / "_views.xml"
    path.write_text(XML)
    model = mujoco.MjModel.from_xml_path(str(path))
    path.unlink(missing_ok=True)

    close = np.clip(sim.BASE_POSE * 1.3,
                    model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    renderer = mujoco.Renderer(model, 380, 500)
    frames, labels = [], []
    for pose, tag in [(np.zeros(model.nu), "张开"), (close, "闭合")]:
        data = mujoco.MjData(model)
        for _ in range(1200):
            data.ctrl[:] = pose
            mujoco.mj_step(model, data)
        for cam, name in [("topZ", "俯视 +Z"), ("frontX", "正视 +X"), ("sideY", "侧视 -Y")]:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
            labels.append(f"{tag} · {name}")
        if tag == "闭合":
            for nm in ["if_ds", "mf_ds", "rf_ds", "th_ds"]:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
                print(f"  闭合时 {nm} 位置 = {np.round(data.xpos[bid], 4)}")

    p = sim.contact_sheet(frames, sim.ROOT / "hand_views.png", cols=3, labels=labels)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
