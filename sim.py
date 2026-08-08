"""公共仿真层：搭场景、把物体正确放到掌心、跑一次抓取 + 拽拉测试。

grasp.py / calibrate.py / sweep.py 都从这里拿逻辑，保证三者口径一致。

坐标系（LEAP 右手，见 hand_views.png）：
  掌心朝 +z，四指从 x≈0 伸向 +x 并向上（+z）卷，拇指在 -y 侧横过来。
  掌面高度 z≈0.129–0.135，所以物体必须放在掌面**上方**，
  不能像最初的 grasp.py 那样直接扔在 z=0.10 —— 那是手掌内部。
"""

import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).parent
HAND_DIR = ROOT / "mujoco_menagerie/leap_hand"  # 资源路径相对此目录，场景须落在这里

G = 9.81
MASS = 0.05          # 所有物体统一质量，保证比的是几何/摩擦而不是重量
WEIGHT = MASS * G    # 0.49 N
FREE_QPOS = 16       # 手有 16 个关节，物体的 freejoint 从 qpos[16] 开始

# 时间轴（秒）
T_OPEN, T_CLOSE, T_HOLD, T_PULL = 0.4, 1.2, 0.8, 6.0
F_MAX = 60.0   # 上拉力最大值 (N)，约 120 倍物体自重；加载速率 10 N/s
SLIP = 0.03    # 相对抓稳位置上移 3cm 即判定被拽走

# grasp.py 那组基准位形：四指弯曲 + 拇指内收
BASE_POSE = np.array([
    0.85, 0.0, 0.95, 0.55,   # 食指 mcp/rot/pip/dip
    0.85, 0.0, 0.95, 0.55,   # 中指
    0.85, 0.0, 0.95, 0.55,   # 无名指
    1.10, 0.55, 0.90, 0.35,  # 拇指 cmc/axl/mcp/ipl
])
JOINT_NAMES = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]

# 掌心里那个最好用的落点，由 calibrate.py 扫出来（见 RESULTS.md）
# x≈4cm 那一整行明显最好（中节指骨正好压在物体上方），y 取靠拇指一侧
BEST_XY = (0.04, 0.02)

# 物体库：统一质量、统一落点，只有几何不同
OBJECTS = {
    "方块 3.0cm":  dict(type="box",       size=".015 .015 .015"),
    "方块 4.4cm":  dict(type="box",       size=".022 .022 .022"),   # 基准
    "方块 5.5cm":  dict(type="box",       size=".0275 .0275 .0275"),
    "方块 6.5cm":  dict(type="box",       size=".0325 .0325 .0325"),
    "球 R2.2cm":   dict(type="sphere",    size=".022"),
    "球 R3.0cm":   dict(type="sphere",    size=".030"),
    "圆柱 立":      dict(type="cylinder",  size=".018 .025"),
    "圆柱 横":      dict(type="cylinder",  size=".018 .025", euler="90 0 0"),
    "细杆 横":      dict(type="cylinder",  size=".008 .060", euler="90 0 0"),
    "椭球":         dict(type="ellipsoid", size=".022 .022 .032"),
}

SCENE_TMPL = """
<mujoco model="leap grasp">
  <include file="right_hand.xml"/>
  <statistic center="0 0 0.14" extent="0.32"/>
  <visual>
    <global azimuth="130" elevation="-20"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".25 .3 .35" rgb2="0 0 0" width="32" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".15 .17 .19" rgb2=".2 .22 .24"
             width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance=".1"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.4 0.7" dir="-0.2 0.5 -1" diffuse="1 1 1"/>
    <light pos="-0.3 0.3 0.6" dir="0.4 -0.4 -1" diffuse=".5 .5 .5"/>
    <geom name="floor" type="plane" size="1 1 .01" material="grid" pos="0 0 -0.02"/>
    <body name="obj" pos="0 0 0.20"{euler}>
      <freejoint/>
      <geom name="obj" type="{type}" size="{size}" rgba=".85 .38 .18 1"
            mass="{mass}" friction="{fric} .01 .001"/>
    </body>
  </worldbody>
</mujoco>
"""


def build(obj_key="方块 4.4cm", fric=1.2, mass=MASS, keep=False):
    """生成场景并编译。keep=True 时保留 xml 方便用 viewer 打开。"""
    spec = OBJECTS[obj_key]
    euler = f' euler="{spec["euler"]}"' if spec.get("euler") else ""
    xml = SCENE_TMPL.format(
        euler=euler, type=spec["type"], size=spec["size"], mass=mass, fric=fric
    )
    path = HAND_DIR / ("_scene_grasp.xml" if keep else "_scene_tmp.xml")
    path.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(path))
    if not keep:
        path.unlink(missing_ok=True)
    return model


def obj_ids(model):
    return (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj"))


def n_obj_contacts(model, data, gid, want_force=False):
    """只数落在被抓物体上的接触点，可选累加法向力。"""
    n, f_total = 0, 0.0
    buf = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == gid or c.geom2 == gid:
            n += 1
            if want_force:
                mujoco.mj_contactForce(model, data, i, buf)
                f_total += buf[0]
    return (n, f_total) if want_force else n


def rest_height(model, data, gid, x, y, quat=(1, 0, 0, 0)):
    """二分找出「刚好不碰到张开的手」的最低高度 —— 物体贴着掌面放，而不是插进去。"""
    def touching(z):
        data.qpos[:] = 0
        data.qpos[FREE_QPOS:FREE_QPOS + 3] = (x, y, z)
        data.qpos[FREE_QPOS + 3:FREE_QPOS + 7] = quat
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        return n_obj_contacts(model, data, gid) > 0

    lo, hi = 0.10, 0.35
    if touching(hi):
        return hi
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if touching(mid) else (lo, mid)
    return hi + 0.001  # 留 1mm，靠重力自己落下去


# 每个条件重复跑的落点扰动 (dx, dy, 绕 z 转的角度)。
# 单次抓取对毫米级差异极敏感，只跑一次得到的数字是运气，不是结论。
JITTERS = [
    (0.000, 0.000, 0.0),
    (0.003, 0.002, +0.20),
    (-0.003, 0.002, -0.20),
    (0.002, -0.003, +0.35),
    (-0.002, -0.002, -0.35),
]


def run(obj_key="方块 4.4cm", fric=1.2, pose=None, xy=None, mass=MASS,
        jitter=(0.0, 0.0, 0.0), pull_dir=(0, 0, 1), shots_at=(), keep_scene=False):
    """张开 → 闭合 → 保持 → 沿 pull_dir 加大拉力拽，返回一组可比的指标。

    jitter:   (dx, dy, yaw) 落点扰动，用来把「运气」摊成分布。
    pull_dir: 拽的方向。+z 是把物体从指笼里拔出去，考的是「包没包住」；
              +y 是沿掌面横推，考的才是摩擦。两个方向答案完全不同。
    shots_at: 需要渲染的时刻（秒）列表，返回 frames。
    """
    model = build(obj_key, fric, mass, keep=keep_scene)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    gid, bid = obj_ids(model)
    zq = FREE_QPOS + 2

    dx, dy, yaw = jitter
    x0, y0 = xy if xy is not None else BEST_XY
    x, y = x0 + dx, y0 + dy
    quat = data.qpos[FREE_QPOS + 3:FREE_QPOS + 7].copy()  # 建模时给的初始姿态
    if not quat.any():
        quat = np.array([1.0, 0, 0, 0])
    if yaw:
        spin = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, spin, quat)
        quat = out
    z0 = rest_height(model, data, gid, x, y, quat)

    mujoco.mj_resetData(model, data)
    data.qpos[FREE_QPOS:FREE_QPOS + 3] = (x, y, z0)
    data.qpos[FREE_QPOS + 3:FREE_QPOS + 7] = quat

    close = np.clip(BASE_POSE if pose is None else pose,
                    model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    open_pose = np.zeros(model.nu)

    n_settle = int((T_OPEN + T_CLOSE + T_HOLD) / dt)
    n_pull = int(T_PULL / dt)

    shot_steps = {int(t / dt): t for t in shots_at}
    renderer = mujoco.Renderer(model, 420, 560) if shots_at else None
    frames, frame_labels = [], []

    # ---- 阶段 1：张开落稳 → 平滑闭合 → 保持 ----
    for step in range(n_settle):
        t = step * dt
        if t < T_OPEN:
            target = open_pose
        elif t < T_OPEN + T_CLOSE:
            a = (t - T_OPEN) / T_CLOSE
            target = open_pose + (3 * a**2 - 2 * a**3) * (close - open_pose)
        else:
            target = close
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
        if step in shot_steps:
            renderer.update_scene(data, camera=-1)
            frames.append(renderer.render())
            frame_labels.append(f"t={shot_steps[step]:.2f}s")

    z_hold = float(data.qpos[zq])
    n_con, f_con = n_obj_contacts(model, data, gid, want_force=True)
    launched = z_hold > z0 + 0.10          # 被指头弹飞
    dropped = z_hold < z0 - 0.05           # 从手里漏下去
    held = not launched and not dropped

    # ---- 阶段 2：沿 pull_dir 匀速加大拉力，记录被拽走时的力 ----
    f_slip = float("nan")
    if held:
        d = np.asarray(pull_dir, float)
        d = d / np.linalg.norm(d)
        p0 = data.qpos[FREE_QPOS:FREE_QPOS + 3].copy()
        for step in range(n_pull):
            f = F_MAX * step / n_pull
            data.xfrc_applied[bid, :3] = f * d
            data.ctrl[:] = close
            mujoco.mj_step(model, data)
            if (data.qpos[FREE_QPOS:FREE_QPOS + 3] - p0) @ d > SLIP:
                f_slip = f
                break
        else:
            f_slip = F_MAX  # 拉满都没拽走
        data.xfrc_applied[bid, :3] = 0.0
        if renderer is not None:
            renderer.update_scene(data, camera=-1)
            frames.append(renderer.render())
            frame_labels.append(f"拽 {f_slip:.1f}N")

    return dict(
        held=held, dropped=dropped, launched=launched,
        z0=z0, z_hold=z_hold, n_con=n_con, f_con=f_con, f_slip=f_slip,
        ratio=(f_slip / WEIGHT if f_slip == f_slip else 0.0),
        frames=frames, frame_labels=frame_labels,
    )


def run_repeats(**kw):
    """把同一个条件在 JITTERS 上各跑一遍，返回 (中位数倍率, 最小, 最大, 失手次数, 明细)。"""
    runs = [run(jitter=j, **kw) for j in JITTERS]
    ratios = [r["ratio"] for r in runs]  # 没抓住的自然是 0
    fails = sum(1 for r in runs if not r["held"])
    return (float(np.median(ratios)), min(ratios), max(ratios), fails, runs)


def band(med, lo, hi, fails):
    """一行显示中位数 + 区间 + 失手次数。"""
    s = f"{med:>6.1f}× [{lo:.1f}–{hi:.1f}]"
    return s + (f"  失手{fails}/{len(JITTERS)}" if fails else "")


def verdict(r):
    """把一次运行压成一行人话。"""
    if r["launched"]:
        return "弹飞"
    if r["dropped"]:
        return "掉了"
    if r["f_slip"] != r["f_slip"]:
        return "—"
    if r["f_slip"] >= F_MAX:
        return f"≥{F_MAX:.0f}N (≥{r['ratio']:.0f}×自重, 拉满未脱)"
    if r["f_slip"] < WEIGHT:
        return f"{r['f_slip']:.2f}N ({r['ratio']:.1f}×自重, 翻手就掉)"
    return f"{r['f_slip']:.2f}N ({r['ratio']:.1f}×自重)"


def _label_font(size=20):
    """PIL 自带字体画不了中文（全是豆腐块），优先用系统里的苹方。"""
    from PIL import ImageFont
    for p in ("/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/Supplemental/Songti.ttc",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(frames, path, cols=3, labels=None):
    from PIL import Image, ImageDraw
    font = _label_font()
    h, w, _ = frames[0].shape
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows))
    for i, img in enumerate(frames):
        im = Image.fromarray(img)
        if labels and i < len(labels):
            dr = ImageDraw.Draw(im)
            dr.text((11, 11), labels[i], fill=(0, 0, 0), font=font)
            dr.text((10, 10), labels[i], fill=(255, 210, 90), font=font)
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(path)
    return path
