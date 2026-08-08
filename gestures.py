"""让 LEAP Hand 做一串复杂手势 —— 从「能不能抓住」进到「能不能摆姿势」。

跑法:
  ./.venv/bin/python gestures.py --sheet   # 只出静态姿势拼图 gestures.png（快，调姿势用）
  ./.venv/bin/python gestures.py           # 出动画 gestures.mp4 + 拼图 + viewer 场景

产物:
  gestures.png  每个手势一格
  gestures.mp4  手势之间平滑过渡的动画（30fps）
  mujoco_menagerie/leap_hand/_scene_gestures.xml
        用 viewer 打开后，左侧面板可以直接切 keyframe 逐个加载手势

注意这只手只有 3 根手指 + 拇指（没有小指），所以「数到 5」做不了，最多到 4。
"""

import argparse
import subprocess

import mujoco
import numpy as np

import sim

# 关节顺序 = sim.JOINT_NAMES
#   食指/中指/无名指 各 4 个: mcp(向掌心弯) rot(左右张开) pip(中节) dip(指尖)
#   拇指 4 个: cmc(抬离掌面) axl(轴向自转) mcp(横过掌心) ipl(指尖)
# 掌心朝 +z，手指从 x≈0 伸向 +x；mcp 越大手指越往 +z 卷起来。


TIPS = ["if_tip", "mf_tip", "rf_tip", "th_tip"]
FINGERS = ["食指", "中指", "无名指"]


def digit(mcp=0.0, rot=0.0, pip=0.0, dip=0.0):
    return [mcp, rot, pip, dip]


def pose(index, middle, ring, thumb):
    return np.array(index + middle + ring + thumb, dtype=float)


STRAIGHT = digit()
CURL = digit(1.55, 0.0, 1.45, 1.10)       # 完全卷进掌心
HALF = digit(0.80, 0.0, 0.90, 0.55)
THUMB_ACROSS = digit(0.55, 1.15, 1.75, 0.70)   # 拇指压在掌心上（握拳时压住四指）
THUMB_UP = digit(1.95, 0.10, 0.00, 0.00)       # 拇指立起来，离开掌面
THUMB_OUT = digit(0.00, 0.00, 0.00, 0.00)

POSES = {
    "张开":     pose(STRAIGHT, STRAIGHT, STRAIGHT, THUMB_OUT),
    "握拳":     pose(CURL, CURL, CURL, THUMB_ACROSS),
    "点赞":     pose(CURL, CURL, CURL, THUMB_UP),
    "指点":     pose(STRAIGHT, CURL, CURL, THUMB_ACROSS),
    "剪刀":     pose(digit(0, -0.45), digit(0, 0.45), CURL, THUMB_ACROSS),
    "数 3":     pose(STRAIGHT, STRAIGHT, STRAIGHT, THUMB_ACROSS),
    "OK":       pose(digit(1.15, 0.15, 0.75, 0.55), STRAIGHT, STRAIGHT,
                     digit(1.05, 0.75, 1.30, 0.25)),
    "捏 · 食指": pose(digit(1.05, 0.20, 0.70, 0.50), STRAIGHT, STRAIGHT,
                     digit(1.00, 0.70, 1.25, 0.30)),
    "捏 · 中指": pose(STRAIGHT, digit(1.05, 0.00, 0.70, 0.50), STRAIGHT,
                     digit(1.15, 0.60, 1.55, 0.30)),
    "捏 · 无名": pose(STRAIGHT, STRAIGHT, digit(1.05, 0.20, 0.70, 0.50),
                     digit(1.25, 0.50, 1.85, 0.30)),
    "张开五指":  pose(digit(0, -0.55), digit(0, 0.0), digit(0, 0.55), digit(0.2, 0, 0, 0)),
    "抓握":     np.clip(sim.BASE_POSE * 1.3, -10, 10),
    "球形抓":   pose(digit(0.75, -0.30, 0.75, 0.45), digit(0.75, 0, 0.75, 0.45),
                     digit(0.75, 0.30, 0.75, 0.45), digit(1.20, 0.60, 0.85, 0.35)),
    "钩":       pose(digit(0.30, 0, 1.60, 1.30), digit(0.30, 0, 1.60, 1.30),
                     digit(0.30, 0, 1.60, 1.30), THUMB_OUT),
}

# 波浪：四个数字依次卷起来再依次放下（墨西哥人浪）
def _wave():
    out = []
    for k in range(3):
        p = pose(STRAIGHT, STRAIGHT, STRAIGHT, THUMB_OUT)
        p[k * 4:k * 4 + 4] = CURL
        out.append((f"波浪 {k + 1}", p))
    out.append(("波浪 · 拇指", pose(STRAIGHT, STRAIGHT, STRAIGHT, THUMB_ACROSS)))
    return out


# 动画顺序：(手势名, 停留秒数)。过渡一律 0.5s。
SEQUENCE = [
    ("张开", 0.6), ("握拳", 0.7), ("张开", 0.4),
    ("指点", 0.6), ("剪刀", 0.6), ("数 3", 0.6), ("张开五指", 0.6),
    ("握拳", 0.5), ("点赞", 0.9), ("张开", 0.4),
    ("捏 · 食指", 0.5), ("张开", 0.25), ("捏 · 中指", 0.5), ("张开", 0.25),
    ("捏 · 无名", 0.5), ("张开", 0.35),
    ("OK", 0.9), ("张开", 0.35),
    ("钩", 0.6), ("张开", 0.3),
    ("球形抓", 0.6), ("抓握", 0.8), ("张开", 0.5),
]
WAVE = _wave()
SEQUENCE += [(n, 0.22) for n, _ in WAVE] + [("张开", 0.3)]
SEQUENCE += [(n, 0.22) for n, _ in reversed(WAVE)] + [("张开", 0.8)]

ALL_POSES = dict(POSES, **dict(WAVE))

TRANSITION = 0.5
FPS = 30

XML = """
<mujoco model="gestures">
  <include file="right_hand.xml"/>
  <statistic center="0 0 0.11" extent="0.30"/>
  <visual>
    <global azimuth="150" elevation="-25" offwidth="720" offheight="900"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".22 .26 .32" rgb2="0 0 0" width="32" height="512"/>
  </asset>
  <worldbody>
    <light pos="0.25 -0.35 0.65" dir="-0.25 0.45 -1" diffuse="1 1 1"/>
    <light pos="-0.25 0.35 0.55" dir="0.3 -0.4 -1" diffuse=".55 .55 .55"/>
    <light pos="0.10 -0.45 0.28" dir="-0.1 1 -0.35" diffuse=".65 .65 .65"/>
    <camera name="palm" pos="0.20 0 0.42" xyaxes="0 -1 0 0.85 0 -0.53"/>
    <camera name="side" pos="0.05 -0.34 0.21" xyaxes="1 0 0 0 0.256 0.967"/>
  </worldbody>
{keys}
</mujoco>
"""


TOUCH_D = 0.024   # 两个指尖 geom 中心相距多远算「刚好贴上」
                  # 指尖不是球是长方体（半边长约 11×12×17mm），没有干净的半径可用


def tip_pos(model, data, name):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return data.geom_xpos[gid].copy()


def tip_dist(model, data, finger):
    return float(np.linalg.norm(tip_pos(model, data, TIPS[finger])
                                - tip_pos(model, data, TIPS[3])))


def solve_pinch(model, finger, restarts=30, seed=0, target=0.018, margin=0.10):
    """求解「拇指指尖去够第 finger 根手指指尖」的关节角。

    只用 mj_forward 做纯运动学（不跑动力学，所以快），
    在该指的 4 个 + 拇指的 4 个关节上做带随机重启的坐标下降，
    目标 target 比「刚好贴上」(TOUCH_D) 略小一点，让指尖互相压一下 ——
    位置伺服是软的，不留压紧量的话稳态误差会把两个指尖分开。
    margin: 关节行程两端各留 10% 不用，给伺服补偿留抬升空间（顶到限位就补不动了）。
    返回 (16 维目标位形, 达到的中心距 m)。
    """
    data = mujoco.MjData(model)
    idx = np.array(list(range(finger * 4, finger * 4 + 4)) + [12, 13, 14, 15])
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    span = hi - lo
    lo, hi = lo + margin * span, hi - margin * span
    rng = np.random.default_rng(seed)

    def cost(q):
        data.qpos[:16] = q
        mujoco.mj_forward(model, data)
        return abs(tip_dist(model, data, finger) - target)

    best_q, best = None, np.inf
    for k in range(restarts):
        q = np.zeros(16)
        # 第 0 次用手写的「捏」当种子，其余随机重启
        q[idx] = (np.array([1.0, 0.2, 0.7, 0.5, 1.1, 0.7, 1.3, 0.3]) if k == 0
                  else lo[idx] + rng.random(8) * (hi[idx] - lo[idx]))
        cur = cost(q)
        for step in (0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.007):
            improved = True
            while improved:
                improved = False
                for j in idx:
                    for s in (+step, -step):
                        trial = q.copy()
                        trial[j] = np.clip(q[j] + s, lo[j], hi[j])
                        v = cost(trial)
                        if v < cur - 1e-6:
                            q, cur, improved = trial, v, True
        if cur < best:
            best_q, best = q.copy(), cur
    data.qpos[:16] = best_q
    mujoco.mj_forward(model, data)
    return best_q, tip_dist(model, data, finger)


def stiffen(model, kp=20.0):
    """把位置伺服调硬一点再做手势。

    默认 kp=3.0 太软，摆出来的姿势和下发的角度差得看得出来。
    实测：kp 从 3 加到 120，稳态角误差一直卡在 ~0.1rad(6°) 不动 ——
    说明这 6° 不是伺服软，是**手指之间自碰撞**顶住了，指令再硬也进不去。
    真机上同理：P 增益调再高也压不掉机构自己的干涉。
    """
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -max(0.01, kp * 0.02)
    return model


def track(model, q_target, iters=10, tol=0.01, gain=0.6):
    """迭代补偿：cmd += gain·(目标 − 实测)，取过程中最好的一次。

    增益取 1.0 会来回震荡（自碰撞让误差非线性），0.6 才收敛。
    返回 (修正后的指令, 稳定后的 data, 最大残余误差 rad)。
    这条对真机一样成立 —— 开环 replay 一组角度，手到不了你要的位置。
    """
    cmd = clip(model, q_target)
    best = (cmd, None, np.inf)
    for _ in range(iters):
        data = settle(model, cmd, steps=900)
        e = q_target - data.qpos[:16]
        err = float(np.abs(e).max())
        if err < best[2]:
            best = (cmd.copy(), data, err)
        if err < tol:
            break
        cmd = clip(model, cmd + gain * e)
    return best


def tip_contacts(model, data):
    """哪两个指尖真的碰在一起了（动力学结果，不是运动学估计）。"""
    ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n): n for n in TIPS}
    out = set()
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in ids and c.geom2 in ids:
            out.add(frozenset((ids[c.geom1], ids[c.geom2])))
    return out


def build(keys=""):
    path = sim.HAND_DIR / "_scene_gestures.xml"
    path.write_text(XML.format(keys=keys))
    return mujoco.MjModel.from_xml_path(str(path)), path


def settle(model, target, steps=1200, data=None):
    """让位置伺服跑到目标角度并稳住。"""
    if data is None:
        data = mujoco.MjData(model)
    for _ in range(steps):
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
    return data


def clip(model, p):
    return np.clip(p, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


# 这两个姿势的关键动作是「离开掌面」，从掌心视角看是正对镜头，拍不出来
CAM_OVERRIDE = {"点赞": "side", "钩": "side", "握拳": "side"}


def resolve(model, verbose=True):
    """把手写/解算出来的目标位形，全部换算成「下发后真能摆出来」的指令。"""
    cmds, worst = {}, []
    for name, p in ALL_POSES.items():
        cmd, _, err = track(model, clip(model, p))
        cmds[name] = cmd
        worst.append((np.degrees(err), name))
    if verbose:
        worst.sort(reverse=True)
        s = "  ".join(f"{n}{e:.1f}°" for e, n in worst[:4])
        print(f"  伺服补偿后残余角误差 最大 4 个: {s}")
    return cmds


def sheet(model, cmds, cam="palm"):
    renderer = mujoco.Renderer(model, 380, 480)
    frames, labels = [], []
    for name, c in cmds.items():
        data = settle(model, c)
        renderer.update_scene(data, camera=CAM_OVERRIDE.get(name, cam))
        frames.append(renderer.render())
        labels.append(name + ("  (侧视)" if name in CAM_OVERRIDE else ""))
    out = sim.contact_sheet(frames, sim.ROOT / "gestures.png", cols=6, labels=labels)
    print(f"-> {out}  ({len(frames)} 个手势)")


def animate(model, cmds, w=640, h=800):
    """按 SEQUENCE 平滑过渡，边跑边渲染，用 ffmpeg 收成 mp4。"""
    dt = model.opt.timestep
    every = max(1, round(1 / FPS / dt))
    renderer = mujoco.Renderer(model, h, w)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         str(sim.ROOT / "gestures.mp4")],
        stdin=subprocess.PIPE,
    )

    data = mujoco.MjData(model)
    prev = cmds["张开"]
    data.ctrl[:] = prev
    step = 0
    for name, hold in SEQUENCE:
        tgt = cmds[name]
        cam = CAM_OVERRIDE.get(name, "palm")   # 立拇指/钩这类动作在掌心视角是正对镜头，看不出来
        for phase, dur in (("move", TRANSITION), ("hold", hold)):
            for _ in range(int(dur / dt)):
                if phase == "move":
                    a = (step % int(TRANSITION / dt)) / (TRANSITION / dt)
                    a = 3 * a**2 - 2 * a**3          # smoothstep，别让伺服硬跳
                    data.ctrl[:] = prev + a * (tgt - prev)
                else:
                    data.ctrl[:] = tgt
                mujoco.mj_step(model, data)
                if step % every == 0:
                    renderer.update_scene(data, camera=cam)
                    proc.stdin.write(renderer.render().tobytes())
                step += 1
        prev = tgt

    proc.stdin.close()
    proc.wait()
    print(f"-> {sim.ROOT / 'gestures.mp4'}  ({step * dt:.1f}s)")


def write_keyframes(model, cmds):
    """把所有手势写成 <key>，viewer 左面板可直接切换加载。"""
    rows = []
    for name, cmd in cmds.items():
        data = settle(model, cmd)
        q = " ".join(f"{v:.3f}" for v in data.qpos[:16])
        c = " ".join(f"{v:.3f}" for v in cmd)
        rows.append(f'    <key name="{name}" qpos="{q}" ctrl="{c}"/>')
    return "  <keyframe>\n" + "\n".join(rows) + "\n  </keyframe>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true", help="只出静态拼图，不渲染视频")
    args = ap.parse_args()

    model = stiffen(build()[0])

    # 「捏」不手写角度，用运动学求解拇指去够每根手指的指尖
    print("求解拇指对指（拇指指尖 → 各手指指尖）:")
    for f in range(3):
        q, d0 = solve_pinch(model, f)
        cmd, data, err = track(model, q)
        touched = bool(tip_contacts(model, data))
        print(f"  {FINGERS[f]:4s} 运动学可达 {d0 * 1000:4.1f}mm | "
              f"落实到动力学后指尖中心距 {tip_dist(model, data, f) * 1000:4.1f}mm | "
              f"{'指尖真的碰上了' if touched else '够不到'}")
        ALL_POSES[f"捏 · {FINGERS[f][:2]}"] = data.qpos[:16].copy()

    # OK 手势 = 解算出来的食指对指 + 中指/无名伸直，不用手写角度
    ok = ALL_POSES["捏 · 食指"].copy()
    ok[4:12] = 0.0
    ALL_POSES["OK"] = ok

    cmds = resolve(model)
    sheet(model, cmds)
    if not args.sheet:
        animate(model, cmds)
        _, path = build(write_keyframes(model, cmds))
        print(f"-> {path}\n   ./.venv/bin/python -m mujoco.viewer "
              f"--mjcf={path.relative_to(sim.ROOT)}")


if __name__ == "__main__":
    main()
