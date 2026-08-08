"""第 1 步 · 批量做实验，回答「抓住到底依赖什么」。

在 grasp.py 的单次抓取之上，把四件事各自扫一遍：
  A. 换东西抓          —— 位形/摩擦不变，只换形状和尺寸
  B. 只改摩擦系数      —— 同一个方块，μ 从 0.05 扫到 3.0
  C. 整体闭合幅度      —— 基准位形 × 系数，从没夹紧到夹爆
  D. 单关节敏感度      —— 16 个关节各 ±0.3rad，看谁真的在出力

评分不看「有没有掉」（掌心朝上，没抓住的东西也会躺在掌心不动），
而是抓稳之后**沿掌面法向往上拽**，看多大的力才把它拽走：
    抓握强度 F_up (N)，以及 F_up / 物体重力 的倍数。
    倍数 ≥ 1 = 把手翻过来也掉不下去 = 真的抓住了，不是托着。

每个条件都在 5 个毫米级扰动的落点上各跑一遍，报中位数和区间 ——
单次抓取对落点极其敏感，只跑一次得到的是运气不是结论。

跑法: ./.venv/bin/python sweep.py   (约 1 分钟)
产物: results.csv / sweep_shapes.png，结论写在 RESULTS.md
"""

import csv

import numpy as np

import sim

ROWS = []


def record(group, label, med, lo, hi, fails, runs, **extra):
    ROWS.append(dict(
        group=group, label=label,
        ratio_median=round(med, 2), ratio_min=round(lo, 2), ratio_max=round(hi, 2),
        fails=fails, n_rep=len(runs),
        n_contact_median=int(np.median([r["n_con"] for r in runs])),
        normal_force_median_N=round(float(np.median([r["f_con"] for r in runs])), 2),
        f_slip_median_N=round(med * sim.WEIGHT, 3),
        **extra,
    ))


def sweep(group, cases, **fixed):
    """cases: [(label, run 关键字参数, 额外记录字段)]"""
    for label, kw, extra in cases:
        med, lo, hi, fails, runs = sim.run_repeats(**{**fixed, **kw})
        record(group, label, med, lo, hi, fails, runs, **extra)
        ncon = int(np.median([r["n_con"] for r in runs]))
        print(f"  {label:<11} 接触{ncon:>3}点   抓握强度 {sim.band(med, lo, hi, fails)}")
        yield label, med, lo, hi, fails, runs


def main():
    print(f"落点 x={sim.BEST_XY[0]:.3f} y={sim.BEST_XY[1]:.3f}（calibrate.py 扫出来的），"
          f"每条件重复 {len(sim.JITTERS)} 次，拉力上限 {sim.F_MAX:.0f}N\n")

    # ---------- A. 形状 ----------
    print("=== A · 换东西抓（位形/摩擦不变）===")
    cases = [(k, dict(obj_key=k), {}) for k in sim.OBJECTS]
    for _ in sweep("A形状", cases):
        pass

    # ---------- B. 摩擦 ----------
    print("\n=== B · 只改摩擦系数（基准方块）===")
    cases = [(f"μ={f}", dict(fric=f), dict(mu=f))
             for f in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 3.0]]
    for _ in sweep("B摩擦", cases):
        pass

    # ---------- C. 整体闭合幅度 ----------
    print("\n=== C · 整体闭合幅度（基准位形 × 系数）===")
    cases = [(f"×{s}", dict(pose=sim.BASE_POSE * s), dict(scale=s))
             for s in [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8]]
    for _ in sweep("C幅度", cases):
        pass

    # ---------- B2. 夹紧之后再看摩擦 ----------
    # B 里摩擦几乎没影响，但那是因为基准位形根本没夹紧（法向力≈0.5N）。
    # 把闭合幅度提到 ×1.5 真的夹住，再扫一遍摩擦，才是摩擦该起作用的场合。
    print("\n=== B2 · 夹紧到 ×1.5 之后，再扫摩擦（还是往上拔）===")
    tight = sim.BASE_POSE * 1.5
    frics = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 3.0]
    cases = [(f"μ={f}", dict(fric=f, pose=tight), dict(mu=f, scale=1.5))
             for f in frics]
    for _ in sweep("B2夹紧后摩擦", cases):
        pass

    # ---------- B3. 换个方向拽：沿掌面横推 ----------
    # 往上拔考的是「指头有没有包过物体顶上」，那是几何，摩擦插不上手。
    # 沿掌面横推没有几何阻挡，能不能推动完全看摩擦 —— 这才是摩擦该出场的地方。
    print("\n=== B3 · 同样夹紧 ×1.5，改成沿掌面横推（+y）===")
    cases = [(f"μ={f}", dict(fric=f, pose=tight, pull_dir=(0, 1, 0)),
              dict(mu=f, scale=1.5, pull="+y")) for f in frics]
    for _ in sweep("B3横推摩擦", cases):
        pass

    # ---------- D. 单关节敏感度 ----------
    print("\n=== D · 每个关节单独 ±0.3rad ===")
    base_med = sim.run_repeats()[0]
    print(f"  基准位形 = {base_med:.1f}×自重\n")
    sens = []
    for j, name in enumerate(sim.JOINT_NAMES):
        got = {}
        for delta in (-0.3, +0.3):
            p = sim.BASE_POSE.copy()
            p[j] += delta
            med, lo, hi, fails, runs = sim.run_repeats(pose=p)
            record("D关节", f"{name}{delta:+.1f}", med, lo, hi, fails, runs,
                   joint=name, delta=delta)
            got[delta] = med
        swing = abs(got[0.3] - got[-0.3])
        sens.append((name, got[-0.3], got[0.3], swing))
        flag = "  ← 敏感" if swing > base_med * 0.4 else ""
        print(f"  {name:<7} -0.3→{got[-0.3]:>6.1f}×   +0.3→{got[0.3]:>6.1f}×"
              f"   落差{swing:>6.1f}{flag}")

    # ---------- 形状拼图 ----------
    shots, labels = [], []
    for k in sim.OBJECTS:
        r = sim.run(obj_key=k, shots_at=[2.35])
        if r["frames"]:
            shots.append(r["frames"][0])
            labels.append(f"{k}  {sim.verdict(r)}")
    if shots:
        p = sim.contact_sheet(shots, sim.ROOT / "sweep_shapes.png", cols=5, labels=labels)
        print(f"\n形状拼图 -> {p}")

    # ---------- 输出 ----------
    out_csv = sim.ROOT / "results.csv"
    keys = ["group", "label", "ratio_median", "ratio_min", "ratio_max", "fails",
            "n_rep", "n_contact_median", "normal_force_median_N", "f_slip_median_N",
            "mu", "scale", "joint", "delta"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(ROWS)
    print(f"明细 -> {out_csv}  ({len(ROWS)} 条)")

    # 谁更能预测抓握强度：接触点数 还是 法向力
    ok = [r for r in ROWS if r["group"] in ("A形状", "B摩擦", "C幅度")]
    n = np.array([r["n_contact_median"] for r in ok], float)
    f = np.array([r["normal_force_median_N"] for r in ok], float)
    y = np.array([r["ratio_median"] for r in ok], float)
    print(f"\n相关性 (n={len(ok)} 个条件)   接触点数 vs 抓握强度 r={np.corrcoef(n, y)[0, 1]:+.2f}"
          f"   |   法向力 vs 抓握强度 r={np.corrcoef(f, y)[0, 1]:+.2f}")

    sens.sort(key=lambda s: -s[3])
    print("最敏感 5 个关节: " + ", ".join(f"{s[0]}({s[3]:.0f})" for s in sens[:5]))
    print("最迟钝 5 个关节: " + ", ".join(f"{s[0]}({s[3]:.1f})" for s in sens[-5:]))


if __name__ == "__main__":
    main()
