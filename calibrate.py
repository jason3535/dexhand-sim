"""标定：把基准方块放在掌心的哪个位置，这只手抓得最牢？

为什么需要这一步：LEAP 手掌面在 z≈0.13，四指从 x≈0 伸向 +x 再往上卷，
拇指从 -y 侧横过来。掌心不同位置，手指能不能形成包络差别极大。
后面所有实验都固定用这里扫出来的最佳落点，否则比较没意义。

跑法: ./.venv/bin/python calibrate.py
产物: calibrate.csv  + 终端里的热力图
"""

import csv

import numpy as np

import sim

XS = [-0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08]
YS = [-0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08]


def main():
    grid, rows = {}, []
    for x in XS:
        for y in YS:
            r = sim.run(xy=(x, y))
            score = 0.0 if not r["held"] else r["ratio"]
            grid[(x, y)] = (score, r)
            rows.append(dict(x=x, y=y, held=int(r["held"]), n_contact=r["n_con"],
                             f_slip_N=(round(r["f_slip"], 3) if r["f_slip"] == r["f_slip"] else ""),
                             ratio_weight=round(r["ratio"], 2)))

    print("抓握强度热力图（数字 = 拽走它需要多少倍自重的力；掉/飞 = 没抓住）")
    print("      y=" + "".join(f"{y*100:>8.0f}" for y in YS) + "   (cm)")
    for x in XS:
        cells = []
        for y in YS:
            s, r = grid[(x, y)]
            if r["dropped"]:
                cells.append("      掉")
            elif r["launched"]:
                cells.append("      飞")
            else:
                cells.append(f"{s:>8.1f}")
        print(f"x={x*100:>5.0f}" + "".join(cells))
    print("      (x 越大越靠指尖，y 越负越靠拇指)")

    best = max(grid, key=lambda k: grid[k][0])
    print(f"\n最高分落点 = x={best[0]:.3f} y={best[1]:.3f}  -> {sim.verdict(grid[best][1])}")

    # 只挑最高分容易挑到「刚好卡住」的运气点：真机上手会抖 1-2cm。
    # 稳健口径 = 周围一圈（±2cm）没有一格是「掉/飞」，再在这些格子里比分数。
    def neighbours(k):
        x, y = k
        return [grid[(round(x + dx, 3), round(y + dy, 3))]
                for dx in (-0.02, 0, 0.02) for dy in (-0.02, 0, 0.02)
                if (round(x + dx, 3), round(y + dy, 3)) in grid]

    safe = [k for k in grid
            if len(neighbours(k)) == 9 and all(nb[1]["held"] for nb in neighbours(k))]
    if safe:
        steady = max(safe, key=lambda k: grid[k][0])
        worst = min(nb[0] for nb in neighbours(steady))
        print(f"最稳落点   = x={steady[0]:.3f} y={steady[1]:.3f}  "
              f"-> {sim.verdict(grid[steady][1])}，周围 ±2cm 全都抓得住，最差 {worst:.1f}×自重")
    print(f"当前 sim.BEST_XY = {sim.BEST_XY}")

    with open(sim.ROOT / "calibrate.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"明细 -> {sim.ROOT / 'calibrate.csv'}")


if __name__ == "__main__":
    main()
