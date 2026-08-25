#!/usr/bin/env bash
# Prove tests/test_leakage.py actually fails when the filter is disabled.
#
# README claims the leakage test "fails when the filter is disabled (verified by mutation, not by
# assertion theatre)". This script is that verification, so the claim is reproducible rather than
# a sentence someone once believed. It mutates a working copy, runs the test, and restores.
set -uo pipefail
cd "$(dirname "$0")/.."

BAK=$(mktemp)
cp src/datamodule.py "$BAK"
restore() { cp "$BAK" src/datamodule.py; rm -f "$BAK"; }
trap restore EXIT

python - <<'PY'
import pathlib
p = pathlib.Path("src/datamodule.py"); t = p.read_text()
old = """        scaf = scaffold_smiles(smi)
        if scaf is None or scaf in banned:
            dropped += 1
            continue
        kept.append(smi)"""
assert t.count(old) == 1, "mutation target not found -- update this script"
p.write_text(t.replace(old, "        kept.append(smi)  # MUTATION: filter disabled"))
PY

python -m pytest tests/test_leakage.py -q > /dev/null 2>&1
status=$?
if [ $status -ne 0 ]; then
  echo "TEETH_OK: leakage test failed with the filter disabled, as it must"
  exit 0
fi
echo "TEETH_FAIL: leakage test PASSED with the filter disabled -- the test is theatre"
exit 1
