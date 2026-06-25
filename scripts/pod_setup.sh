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

say "3/5  install project deps + DeepSpeed (let pip resolve trl/transformers against THIS torch)"
# Do NOT pin torch here — the whole point of a clean image is that its torch is already right.
# We install [l3] (NOT [infer]: vLLM under multi-GPU DeepSpeed GRPO is a separate, harder
# integration — the --scale run uses HF generate / --no-vllm). DeepSpeed is needed for ZeRO.
# If the resolver tries to *replace* torch, that's the signal the image is wrong — stop.
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
pip install --no-cache-dir -e ".[l3]" deepspeed
TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || die "pip changed torch ($TORCH_BEFORE -> $TORCH_AFTER) — the image's torch was incompatible with current trl/deepspeed. Pick a template matched to the deps, or pin trl/transformers to torch $TORCH_BEFORE's era."

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

say "5/5  single-GPU smoke (0.5B, 3 seeds) — proves the loop + pre-downloads weights/data"
python -m verge.l3_reasoner.run --engine grpo --smoke

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "visible GPUs: ${NGPU}"

cat <<NEXT

=== setup OK (${NGPU} GPUs visible) ===
The single-GPU 0.5B smoke proved the loop. Next, validate the MULTI-GPU plumbing cheaply
(0.5B across all GPUs via DeepSpeed ZeRO-2) BEFORE spending 7B hours:

  accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes ${NGPU} \\
      -m verge.l3_reasoner.run --engine grpo --smoke

If that prints ONE clean slope+CI report (not ${NGPU} copies), rank-gating + ZeRO work.
Then the real §5.4 headroom run (full-FT 7B):

  accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes ${NGPU} \\
      -m verge.l3_reasoner.run --engine grpo --scale --dataset gsm8k --no-vllm

Notes:
  - --scale = 7B, K=16, 4 rounds, 3 seeds, 800/500 splits, 150 steps/round, grad-checkpointing.
  - Keep (per_device_batch * num_processes) a multiple of K=16. --scale sets per_device=4
    (good for 4 GPUs). For ${NGPU} GPUs, pass e.g. --per-device-batch \$((16/${NGPU})) if 16%${NGPU}==0.
  - ZeRO-2 keeps a full param replica per GPU so eval generation works; 4x80GB comfortable,
    2x80GB tight (add offload_optimizer_device: cpu to the yaml, or use more GPUs).
  - --dataset math = NUMERIC SLICE only (the EXACT verifier's reach); general MATH needs the
    sympy MathEquivalenceRung (not yet built).
  - vLLM is intentionally NOT installed: vLLM + multi-GPU + DeepSpeed GRPO is a separate
    integration. --no-vllm (HF generate) is correct for this run.
NEXT
