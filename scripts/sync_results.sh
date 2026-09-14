#!/usr/bin/env bash
# Pull finished runs off dgx2 into ./results, and say what is still only remote.
#
# Only n100 result files come across. Smoke tests, dev-era probes and the
# `corrupt/` directory stay on the box: they match the arm's benchmark field, so
# collate.unclaimed() would list every one of them and bury the real warnings.
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE=${REMOTE:-dgx2}
RDIR=/raid/nbe_tmp/markus_frey/asp/results

for pat in '*_n100.json'; do
  for try in 1 2 3; do
    rsync -q -e "ssh -o ConnectTimeout=20" "$REMOTE:$RDIR/$pat" results/ 2>/dev/null && break
    echo "  rsync attempt $try failed (dgx2 sshd is intermittent), retrying"; sleep 10
  done
done

echo "local n100 result files: $(ls results/*_n100.json 2>/dev/null | wc -l | tr -d ' ')"
ssh -o ConnectTimeout=20 "$REMOTE" "ls $RDIR/*_n100.json 2>/dev/null | wc -l" </dev/null \
  | sed 's/^/remote n100 result files: /'
echo
echo "in flight on $REMOTE:"
ssh -o ConnectTimeout=20 "$REMOTE" 'pgrep -af "evaluate_vlm.py" | grep -o "benchmark [^ ]* --model [^ ]*" || echo "  nothing running"' </dev/null
