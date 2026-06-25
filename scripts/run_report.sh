#!/usr/bin/env bash
# Run a command, capture ALL output to a persistent log, and post that log to a Discord
# webhook so the run is never lost (even if the pod dies or the terminal scrolls away).
#
# The webhook URL is read from $DISCORD_WEBHOOK — it is NOT stored in this repo (anything
# committed to git is effectively public; a leaked webhook lets anyone spam your channel).
# Set it once per shell:
#   export DISCORD_WEBHOOK='https://discord.com/api/webhooks/...'
#   bash scripts/run_report.sh python -m verge.l3_reasoner.run --engine grpo --budget --dryrun --dataset gsm8k
set -uo pipefail
: "${DISCORD_WEBHOOK:?set DISCORD_WEBHOOK first: export DISCORD_WEBHOOK='<your webhook url>'}"

ts=$(date +%Y%m%d-%H%M%S)
log="/workspace/verge_${ts}.log"
echo "logging to $log"

# run, stream to terminal AND file
"$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

# post the whole log as a file attachment + a one-line status (Discord text caps at 2000 chars)
tail_msg=$(tail -c 1500 "$log" | sed 's/"/\\"/g' | tr '\n' ' ')
curl -sS \
  -F "payload_json={\"content\":\"\`verge ${ts}\` exited **${status}** — tail: ${tail_msg}\"}" \
  -F "file=@${log}" \
  "$DISCORD_WEBHOOK" >/dev/null \
  && echo "posted log to Discord ($log)" \
  || echo "WARN: failed to post to Discord (run still completed; log at $log)"

exit "$status"
