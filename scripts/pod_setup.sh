#!/usr/bin/env bash
# Clean-pod setup for the L3 GRPO headroom run (spec §5.4).
#
# Encodes the hard-won lessons from the A40 shakeout so the next pod doesn't repeat the
# downgrade roulette. Run it on a FRESH pod whose image already ships torch >= 2.6 on a
# CUDA >= 12.6 driver (RunPod "PyTorch 2.6/2.7 CUDA 12.x" templates). Each step gates the
# next: if a check fails the script stops and tells you why.
#
#   bash scripts/pod_setup.sh            # verify -> install -> import-check -> smoke
#
# Then the real run (see the bottom of this script):
#   python -m verge.l3_reasoner.run --engine grpo --scale --dataset gsm8k
set -euo pipefail

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nABORT: %s\n' "$*" >&2; exit 1; }

say "1/5  GPU + torch sanity"
python - <<'PY' || die "torch can't see the GPU — pick a template whose driver matches its torch (this was the #1 time-sink on the A40)."
import sys, torch
ok = torch.cuda.is_available()
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
print(f"torch {torch.__version__}  cuda_available={ok}  device={torch.cuda.get_device_name(0) if ok else 'N/A'}")
if not ok:
    sys.exit(1)
# trl >= 0.16 (full DAPO: clip-higher, etc.) needs torch >= 2.6 for torch.distributed.fsdp.FSDPModule
if v < (2, 6):
    print(f"WARNING: torch {v} < 2.6 — you'll be capped at trl<=0.15 (PLAIN GRPO, no DAPO clip-higher).")
PY

say "2/5  disk headroom"
# The A40 pod died at [Errno 28] with ~4 GB free. 7B weights + deps need room.
avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "root free: ${avail_gb} GB"
[ "${avail_gb:-0}" -ge 30 ] || die "only ${avail_gb} GB free; 7B + CUDA wheels need ~30 GB. Use a bigger volume or set HF_HOME to one."

say "3/5  install project deps (let pip resolve trl/transformers/vllm against THIS torch)"
# Do NOT pin torch here — the whole point of a clean image is that its torch is already right.
# We install [l3,infer]; if the resolver tries to *replace* torch, that's the signal the
# image is wrong — stop and pick a better template rather than fighting it.
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
pip install --no-cache-dir -e ".[l3,infer]"
TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || die "pip changed torch ($TORCH_BEFORE -> $TORCH_AFTER) — the image's torch was incompatible with current trl/vllm. Pick a template matched to the deps, or pin trl/transformers to torch $TORCH_BEFORE's era."

say "4/5  import-check the bridge (no torchaudio / FSDPModule / sampler traps)"
python - <<'PY' || die "bridge import failed — paste the traceback. Known traps: transformers 5.x imports torchaudio (cu mismatch); trl 0.16+ needs torch>=2.6; transformers 4.57 changed _get_train_sampler signature."
import torch, transformers, trl
from trl import GRPOConfig, GRPOTrainer
print(f"OK  trl {trl.__version__}  transformers {transformers.__version__}  torch {torch.__version__}")
try:
    import vllm
    print(f"vllm {vllm.__version__} importable")
except Exception as e:
    print(f"vllm NOT importable ({type(e).__name__}) — run with --no-vllm; HF-generate still works, just slower.")
PY

say "5/5  GPU smoke (0.5B, 3 seeds) — proves the loop before spending 7B hours"
python -m verge.l3_reasoner.run --engine grpo --smoke

cat <<'NEXT'

=== setup OK ===
The 0.5B smoke proved the loop end-to-end on this pod. For the §5.4 headroom run:

  # vLLM path (fast generation) — only if step 4 said "vllm ... importable":
  python -m verge.l3_reasoner.run --engine grpo --scale --dataset gsm8k

  # safe fallback if vLLM/trl are mismatched (trl historically caps at vLLM 0.19):
  python -m verge.l3_reasoner.run --engine grpo --scale --dataset gsm8k --no-vllm

Notes:
  - --scale = 7B, K=16, 4 rounds, 3 seeds, 800/500 splits, 150 steps/round.
  - --dataset math runs the NUMERIC SLICE only (the EXACT verifier's reach). General MATH
    (fractions/surds/symbolic) needs the sympy MathEquivalenceRung — not yet built.
  - With torch>=2.6 + trl>=0.16, GRPOSettings' DAPO knobs (epsilon_high clip-higher,
    scale_rewards, beta=0) now actually take effect (on trl 0.15 they were filtered out).
NEXT
