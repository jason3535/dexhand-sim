#!/usr/bin/env bash
# 从零重建运行环境。仓库里没有 .venv/ 和 mujoco_menagerie/（一共 135M），
# 这个脚本把它们拉回来。macOS / Apple Silicon 上验证过。
set -euo pipefail
cd "$(dirname "$0")"

# mujoco_menagerie 是官方模型库，一整个仓库 3G+，只稀疏拉三只手。
# 固定在 c1a4eeb —— 所有实验结论都是在这个版本的模型上跑出来的，
# 换 commit 数字可能对不上。
MENAGERIE_COMMIT=c1a4eeb

if [ ! -d mujoco_menagerie ]; then
  echo "==> 拉 mujoco_menagerie（稀疏，只要 leap_hand / shadow_hand / wonik_allegro）"
  git clone --filter=blob:none --no-checkout \
    https://github.com/google-deepmind/mujoco_menagerie.git
  git -C mujoco_menagerie sparse-checkout init --cone
  git -C mujoco_menagerie sparse-checkout set leap_hand shadow_hand wonik_allegro
  git -C mujoco_menagerie checkout "$MENAGERIE_COMMIT"
else
  echo "==> mujoco_menagerie 已存在，跳过"
fi

if [ ! -d .venv ]; then
  echo "==> 建 Python 3.12 venv + 装 MuJoCo"
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python mujoco numpy pillow
else
  echo "==> .venv 已存在，跳过"
fi

echo "==> 自检"
./.venv/bin/python check.py

cat <<'EOF'

环境好了。接下来：
  ./.venv/bin/python grasp.py      # 抓一个方块
  ./.venv/bin/python gestures.py   # 18 个手势 + 动画
先读 RESULTS.md。
EOF
