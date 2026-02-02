#!/bin/bash
set -euo pipefail

# Make sure `dump_bash_state` (a cluster/launcher hook) is resolvable from PATH.
# We provide a no-op implementation at /finance_ML/fengninghui/bin/dump_bash_state.
export PATH="/finance_ML/fengninghui/bin:${PATH}"

# Optional: conda helper (your conda is here)
if [ -f "/finance_ML/fengninghui/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/finance_ML/fengninghui/miniconda3/etc/profile.d/conda.sh"
fi

echo "[finai_env] PATH prefixed with /finance_ML/fengninghui/bin"

# 每次跑 srun 之前先执行一次：
# source /finance_ML/fengninghui/LatentMAS/scripts/finai_env.sh
# 然后再正常 srun ...，这样退出时调用 dump_bash_state 就不会再 command not found，也不会把退出码弄成 127。
# 也可以不用 sou

# 也可以不用 source，直接一行解决
# 把 PATH 前缀写到命令前面：
# PATH=/finance_ML/fengninghui/bin:$PATH srun ...

