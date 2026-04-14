#!/bin/bash
# Run all three experiment pipelines sequentially.
# Usage: bash run_all.sh

PYTHON=/home/xujiale/anaconda3/envs/vinfo/bin/python3.10
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo " Starting all pipelines"
echo " Python: $PYTHON"
echo " Working dir: $SCRIPT_DIR"
echo " Start time: $(date)"
echo "============================================================"

run_pipeline() {
    local name="$1"
    local script="$2"
    echo ""
    echo "------------------------------------------------------------"
    echo " [$name] Starting: $script"
    echo " Time: $(date)"
    echo "------------------------------------------------------------"
    "$PYTHON" "$SCRIPT_DIR/$script"
    local code=$?
    if [ $code -eq 0 ]; then
        echo "------------------------------------------------------------"
        echo " [$name] FINISHED OK  (exit 0)  $(date)"
        echo "------------------------------------------------------------"
    else
        echo "------------------------------------------------------------"
        echo " [$name] FAILED  (exit $code)  $(date)"
        echo "------------------------------------------------------------"
    fi
    return $code
}

cd "$SCRIPT_DIR"

run_pipeline "DF"          "experiment/pipeline.py"
run_pipeline "DF-Vinfo"    "experiment/pipeline_vinfo.py"
run_pipeline "Baseline"    "experiment/pipeline_baseline.py"

echo ""
echo "============================================================"
echo " All pipelines done.  End time: $(date)"
echo "============================================================"
