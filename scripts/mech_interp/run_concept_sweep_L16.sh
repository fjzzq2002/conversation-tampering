#!/usr/bin/env bash
# 5 test concepts (Lovely / Sure / Assistant / Bread / Justice) compared against
# 30 random-noun controls at inject layer 16, sweeping strengths 0.005..0.5.
# Assumes the calling shell has Python with torch/transformers in PATH and
# (optionally) HF_HOME set to a local model cache.
set -euo pipefail
cd "$(dirname "$0")"

CONTROLS=(Cat Blue Tree Run Apple Light Stone River Book Cloud Luminous Gravel Whisper Alloy Mirage Tundra Velvet Orbit Lantern Mosaic Entropy Liminality Resonance Friction Emergence Decay Symmetry Drift Abstraction Saturation)
STRENGTHS=(0.005 0.01 0.02 0.03 0.05 0.1 0.2 0.5)

run_one () {
  local CONCEPT="$1" TOKEN="$2" USERPROMPT="$3"
  local TAG; TAG="$(echo "$CONCEPT" | tr '[:upper:]' '[:lower:]')"
  python -u concept_inj_compare.py \
    --concept "$CONCEPT" "${CONTROLS[@]}" \
    --inject-token "$TOKEN" \
    --user "$USERPROMPT" \
    --asst-tail "The sky is clear today." \
    --strength "${STRENGTHS[@]}" \
    --inject-at all \
    --inject-layer 16 \
    --out "concept_inj_full_${TAG}_L16.json"
}

run_one Lovely    "Lovely."    "Say a sentence without 'lovely.'"
run_one Sure      "Sure."      "Say a sentence without 'sure.'"
run_one Assistant "Assistant:" "Say a sentence without 'assistant.'"
run_one Bread     "Bread."     "Say a sentence without 'bread.'"
run_one Justice   "Justice."   "Say a sentence without 'justice.'"

echo "all done"
