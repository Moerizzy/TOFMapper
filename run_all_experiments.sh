#!/usr/bin/env bash
# Train + test all multimodal TOF configs sequentially.
#
# Runs each experiment one after another:
#   1. RGB only              (3 bands)
#   2. RGB + NIR             (4 bands)
#   3. RGB + nDSM            (4 bands)
#   4. RGB + NIR + nDSM      (5 bands)
#   5. RGB with class-aware oversampling on Patch (ablation)
#
# After each training, evaluates the best checkpoint on the test set and
# saves metrics + confusion matrices under runs/<run_tag>/test/.
#
# All training logs go to runs/<run_tag>/train.log; all test logs to
# runs/<run_tag>/test.log.
#
# Usage:
#   bash run_all_experiments.sh                # run everything
#   bash run_all_experiments.sh --dry-run      # only print commands
#   bash run_all_experiments.sh rgb rgbn       # only the named tags
#   SKIP_TRAIN=1 bash run_all_experiments.sh   # skip training, only test
#   SKIP_TEST=1  bash run_all_experiments.sh   # skip testing, only train
#   TTA=d4 bash run_all_experiments.sh         # run test with d4 TTA

set -euo pipefail

# This script lives inside the repo (TOFMapper/), but the data and the
# 'TOFMapper/...' invocation paths are anchored ONE LEVEL UP. So we cd to
# the parent of the script's directory and use 'TOFMapper/...' prefixes
# below — this matches how the other tools are invoked (e.g.
# `python TOFMapper/tools/build_5band_stack.py --state BB`).
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$(dirname "$SCRIPT_DIR")"

REPO="TOFMapper"

# --- Config registry (run_tag : config_path) -------------------------------
declare -a RUN_TAGS=(
    "rgb"
    "rgbn"
    "rgb_ndsm"
    "rgbn_ndsm"
    "rgb_oversample"
)

declare -A CONFIG_FOR
CONFIG_FOR[rgb]="$REPO/config/tof/ortho_all_regions_rgb.py"
CONFIG_FOR[rgbn]="$REPO/config/tof/ortho_all_regions_4bands_norm.py"
CONFIG_FOR[rgb_ndsm]="$REPO/config/tof/ortho_all_regions_rgb_ndsm.py"
CONFIG_FOR[rgbn_ndsm]="$REPO/config/tof/ortho_all_regions_rgbn_ndsm.py"
CONFIG_FOR[rgb_oversample]="$REPO/config/tof/ortho_all_regions_rgb_oversample.py"

# --- Parse args ------------------------------------------------------------
DRY_RUN=0
SELECTED=()
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n)  DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)             SELECTED+=("$arg") ;;
    esac
done
if [[ ${#SELECTED[@]} -eq 0 ]]; then
    SELECTED=("${RUN_TAGS[@]}")
fi

SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_TEST=${SKIP_TEST:-0}
TTA=${TTA:-}

run() {
    echo "+ $*"
    if [[ $DRY_RUN -eq 0 ]]; then
        "$@"
    fi
}

OUTPUT_BASE="runs"

for tag in "${SELECTED[@]}"; do
    cfg="${CONFIG_FOR[$tag]:-}"
    if [[ -z "$cfg" ]]; then
        echo "ERROR: unknown run tag '$tag'. Known tags: ${RUN_TAGS[*]}"
        exit 1
    fi
    if [[ ! -f "$cfg" ]]; then
        echo "ERROR: config not found at $cfg"
        exit 1
    fi

    run_dir="$OUTPUT_BASE/$tag"
    train_log="$run_dir/train.log"
    test_log="$run_dir/test.log"
    test_out="$run_dir/test"
    [[ $DRY_RUN -eq 0 ]] && mkdir -p "$run_dir" "$test_out"

    echo
    echo "=========================================================="
    echo "  [$tag]   config = $cfg"
    echo "  log dir = $run_dir"
    echo "=========================================================="

    # ---- train -----------------------------------------------------------
    if [[ "$SKIP_TRAIN" == "0" ]]; then
        echo ">>> Training $tag (log: $train_log)"
        if [[ $DRY_RUN -eq 0 ]]; then
            python $REPO/train_supervision.py -c "$cfg" 2>&1 | tee "$train_log"
        else
            run python $REPO/train_supervision.py -c "$cfg"
        fi
    else
        echo ">>> SKIP_TRAIN=1 — skipping training for $tag"
    fi

    # ---- test ------------------------------------------------------------
    if [[ "$SKIP_TEST" == "0" ]]; then
        echo ">>> Testing $tag (log: $test_log)"
        tta_arg=()
        if [[ -n "$TTA" ]]; then
            tta_arg=(-t "$TTA")
        fi
        if [[ $DRY_RUN -eq 0 ]]; then
            python $REPO/tof_test.py -c "$cfg" -o "$test_out" "${tta_arg[@]}" \
                2>&1 | tee "$test_log"
        else
            run python $REPO/tof_test.py -c "$cfg" -o "$test_out" "${tta_arg[@]}"
        fi
    else
        echo ">>> SKIP_TEST=1 — skipping test for $tag"
    fi
done

echo
echo "=========================================================="
echo "  All experiments finished."
echo "  Per-run results: $OUTPUT_BASE/<tag>/test/"
echo "    - metrics_<region>.json      (per-class P/R/F1/IoU/Dice)"
echo "    - normalized_error_matrix_<region>.csv"
echo "    - overall_normalized_error_matrix.csv"
echo "=========================================================="
