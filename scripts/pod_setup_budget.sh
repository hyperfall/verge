#!/usr/bin/env bash
# Single-A100 7B headroom run on a TIGHT budget (~$13.62 ≈ 10 A100-GPU-hours).
#
# Path: ONE 80GB A100, full-FT 7B via vLLM colocate (fast generation) + 8-bit Adam + bf16
# weights + grad-checkpointing. No DeepSpeed, no accelerate — plain `python -m`, the proven
# single-process harness. The risk is the trl<->vLLM version match; this script gates it with
# a ~$0.50 dry run BEFORE the real spend. Watch the dollar notes — there is no debug buffer.
#
#   bash scripts/pod_setup_budget.sh        # verify -> install -> import -> DRY RUN (gate)
# then, only if the dry run is clean:
#   python -m verge.l3_reasoner.run --engine grpo --budget --dataset gsm8k
set -euo pipefail
say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nABORT: %s\n' "$*" >&2; exit 1; }

say "1/4  GPU + torch + disk (each costs money from here — fail fast)"
python - <<'PY' || die "torch can't see the GPU, or it's not an 80GB card. A 7B full-FT + vLLM needs one 80GB A100."
import torch
assert torch.cuda.is_available(), "no CUDA"
name = torch.cuda.get_device_name(0)
gb = torch.cuda.get_device_properties(0).total_memory / 1e9
v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
print(f"torch {torch.__version__}  {name}  {gb:.0f} GB")
assert gb >= 70, f"GPU has only {gb:.0f} GB — 7B+vLLM needs ~80 GB"
if v < (2, 6):
    print(f"WARNING: torch {v} < 2.6 — trl will be <=0.15 (plain GRPO, no DAPO clip-higher).")
PY
avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "root free: ${avail} GB"
[ "${avail:-0}" -ge 25 ] || die "only ${avail} GB free; 7B weights (~15GB) + deps need ~25GB. Set HF_HOME to a bigger mount or resize."

say "2/4  install [l3,infer] (vLLM) + bitsandbytes (8-bit Adam)"
TB=$(python -c "import torch; print(torch.__version__)")
pip install --no-cache-dir -e ".[l3,infer]" bitsandbytes
TA=$(python -c "import torch; print(torch.__version__)")
[ "$TB" = "$TA" ] || die "pip changed torch ($TB -> $TA) — image torch is incompatible with vllm/trl. Pick a template matched to the deps."

say "3/4  import-check: trl bridge + vLLM + bitsandbytes all importable"
python - <<'PY' || die "import failed — paste the traceback. The usual culprit is a trl<->vLLM version mismatch (trl caps the vLLM range it supports)."
import torch, transformers, trl, vllm, bitsandbytes
from trl import GRPOConfig, GRPOTrainer
print(f"OK  trl {trl.__version__}  transformers {transformers.__version__}  vllm {vllm.__version__}  torch {torch.__version__}")
# sanity: the GRPOConfig knobs --budget relies on actually exist in this trl
import inspect
p = set(inspect.signature(GRPOConfig.__init__).parameters)
for k in ("use_vllm", "vllm_mode", "vllm_gpu_memory_utilization", "optim"):
    print(f"  GRPOConfig has {k}: {k in p}")
PY

say "4/4  DRY RUN gate (~\$0.50): proves 7B + vLLM colocate + 8-bit Adam load & step"
# 3 seeds (the §7 guard) but 3 steps + 16-example evals — trivially short.
python -m verge.l3_reasoner.run --engine grpo --budget --dryrun --dataset gsm8k

cat <<'NEXT'

=== dry run passed ===
The toolchain works (7B loaded, vLLM colocate generated, 8-bit Adam stepped). NOW the real run:

  python -m verge.l3_reasoner.run --engine grpo --budget --dataset gsm8k

Budget notes (single A100 @ ~$1.39/hr):
  - --budget = 7B, K=8, 3 rounds, 60 steps/round, 3 seeds, 400/200 splits. Est ~2-3 h ≈ $3-5.
  - Watch GPU memory on the FIRST few steps (`nvidia-smi`). If it OOMs, lower vLLM's share:
        ... --budget --vllm-mem 0.2        (gives the policy/optimizer more room)
    or drop K: ... --budget --k 4
  - If vLLM itself won't cooperate, fall back to HF generate (slower, but no vLLM):
        ... --budget --no-vllm             (then expect a longer / pricier run — watch the clock)
  - This is the real §5.4 attempt, but REDUCED (K=8, 180 total steps). Report it honestly as a
    budget-constrained headroom probe, not the full K=16/4-round result.
NEXT
