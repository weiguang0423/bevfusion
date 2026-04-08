#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_INPUT=${1:-"projects/BEVFusion/configs/bevfusion_lidar-cam-radar_geometry_enhanced_stage2_e12.py"}
WORK_DIR_INPUT=${2:-"work_dirs/bevfusion_radar_weather_stage2_final_e12_cameraaware_epoch1"}
GPUS=${3:-2}
MAX_EPOCH=${4:-12}
PYTHON_BIN=${PYTHON_BIN:-/home/vipuser/miniconda3/envs/bevfusion/bin/python}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
WEATHER_TEST_WORKERS=${WEATHER_TEST_WORKERS:-2}
WEATHER_TEST_PERSISTENT_WORKERS=${WEATHER_TEST_PERSISTENT_WORKERS:-False}
WEATHER_TEST_PIN_MEMORY=${WEATHER_TEST_PIN_MEMORY:-False}
WEATHER_TEST_PREFETCH_FACTOR=${WEATHER_TEST_PREFETCH_FACTOR:-2}

resolve_repo_path() {
    local input_path="$1"
    if [[ "$input_path" = /* ]]; then
        realpath -m "$input_path"
    else
        realpath -m "$ROOT_DIR/$input_path"
    fi
}

CONFIG=$(resolve_repo_path "$CONFIG_INPUT")
WORK_DIR=$(resolve_repo_path "$WORK_DIR_INPUT")
WEATHER_DIR="$WORK_DIR/weather_eval"
ANN_DIR="$WEATHER_DIR/ann_splits"
RUNNER_DIR="$WORK_DIR/epochwise_runner"
SUMMARY_FILE="$RUNNER_DIR/weather_eval_summary.tsv"
FULL_SUMMARY_FILE="$RUNNER_DIR/full_eval_summary.tsv"

mkdir -p "$RUNNER_DIR"

log() {
    local message="$1"
    printf '[%s] %s\n' "$(date '+%F %T')" "$message" | tee -a "$RUNNER_DIR/supervisor.log"
}

current_epoch() {
    local checkpoint_path checkpoint_name
    if [[ ! -f "$WORK_DIR/last_checkpoint" ]]; then
        printf '0'
        return 0
    fi

    checkpoint_path=$(<"$WORK_DIR/last_checkpoint")
    checkpoint_name=$(basename "$checkpoint_path")
    if [[ "$checkpoint_name" =~ ^epoch_([0-9]+)\.pth$ ]]; then
        printf '%d' "${BASH_REMATCH[1]}"
        return 0
    fi

    printf '0'
}

require_ann_splits() {
    local group ann_file
    for group in day night rain; do
        ann_file="$ANN_DIR/nuscenes_infos_val_${group}.pkl"
        if [[ ! -f "$ann_file" ]]; then
            log "missing ann split: $ann_file"
            return 1
        fi
    done
}

extract_metric() {
    local log_file="$1"
    local metric_name="$2"
    local value

    value=$(grep -Eo "${metric_name}: [0-9]+\.[0-9]+" "$log_file" | tail -n 1 | awk '{print $2}') || true
    printf '%s' "$value"
}

build_test_cfg_options() {
    local ann_file="$1"
    local num_workers="$2"
    local persistent_workers="$3"
    local pin_memory="$4"
    local prefetch_factor="$5"

    TEST_CFG_OPTIONS=(
        test_dataloader.dataset.ann_file="$ann_file"
        test_evaluator.ann_file="$ann_file"
        test_dataloader.num_workers="$num_workers"
        test_dataloader.persistent_workers="$persistent_workers"
        test_dataloader.pin_memory="$pin_memory"
    )

    if (( num_workers > 0 )); then
        TEST_CFG_OPTIONS+=(test_dataloader.prefetch_factor="$prefetch_factor")
    fi
}

record_summary() {
    local target_epoch="$1"
    local group="$2"
    local nds="$3"
    local map="$4"
    local tmp_file

    tmp_file=$(mktemp)
    if [[ -f "$SUMMARY_FILE" ]]; then
        grep -Ev "^${target_epoch}[[:space:]]+${group}[[:space:]]+" "$SUMMARY_FILE" > "$tmp_file" || true
    fi
    printf '%s\t%s\t%s\t%s\n' "$target_epoch" "$group" "${nds:-NA}" "${map:-NA}" >> "$tmp_file"
    mv "$tmp_file" "$SUMMARY_FILE"
}

record_full_summary() {
    local target_epoch="$1"
    local nds="$2"
    local map="$3"
    local tmp_file

    tmp_file=$(mktemp)
    if [[ -f "$FULL_SUMMARY_FILE" ]]; then
        grep -Ev "^${target_epoch}[[:space:]]+" "$FULL_SUMMARY_FILE" > "$tmp_file" || true
    fi
    printf '%s\t%s\t%s\n' "$target_epoch" "${nds:-NA}" "${map:-NA}" >> "$tmp_file"
    mv "$tmp_file" "$FULL_SUMMARY_FILE"
}

weather_eval_completed() {
    local target_epoch="$1"
    local group="$2"
    local outdir
    local runner_log

    printf -v outdir '%s/external_epoch_%03d_2gpu_%s' "$WEATHER_DIR" "$target_epoch" "$group"
    runner_log="$outdir/runner.log"

    [[ -f "$runner_log" ]] && grep -q "=== DONE ${group} 2GPU ===" "$runner_log"
}

run_train_epoch() {
    local target_epoch="$1"
    local train_log="$RUNNER_DIR/train_to_epoch_${target_epoch}.log"
    local train_port=$((29540 + target_epoch))
    local train_args=()
    local full_nds
    local full_map

    log "resume training to epoch ${target_epoch}"
    if [[ -f "$WORK_DIR/last_checkpoint" ]]; then
        train_args+=(--resume)
    fi

    (
        export PATH="$(dirname "$PYTHON_BIN"):$PATH"
        export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
        export PORT="$train_port"
        bash "$ROOT_DIR/tools/dist_train.sh" "$CONFIG" "$GPUS" \
            --work-dir "$WORK_DIR" \
            "${train_args[@]}" \
            --cfg-options train_cfg.max_epochs="$target_epoch"
    ) | tee "$train_log"

    if [[ ! -f "$WORK_DIR/epoch_${target_epoch}.pth" ]]; then
        log "missing checkpoint after training: $WORK_DIR/epoch_${target_epoch}.pth"
        return 1
    fi

    full_nds=$(extract_metric "$train_log" 'NDS')
    full_map=$(extract_metric "$train_log" 'mAP')
    log "epoch=${target_epoch} full_eval NDS=${full_nds:-NA} mAP=${full_map:-NA}"
    record_full_summary "$target_epoch" "${full_nds:-NA}" "${full_map:-NA}"
}

run_weather_eval() {
    local target_epoch="$1"
    local group="$2"
    local group_index="$3"
    local checkpoint="$WORK_DIR/epoch_${target_epoch}.pth"
    local ann_file="$ANN_DIR/nuscenes_infos_val_${group}.pkl"
    local outdir
    local test_log
    local test_port
    local nds
    local map
    local attempt
    local num_workers
    local persistent_workers
    local pin_memory
    local prefetch_factor

    printf -v outdir '%s/external_epoch_%03d_2gpu_%s' "$WEATHER_DIR" "$target_epoch" "$group"
    test_log="$outdir/test_stdout.log"
    test_port=$((29640 + target_epoch * 10 + group_index))

    rm -rf "$outdir"
    mkdir -p "$outdir"
    echo "=== RUN ${group} 2GPU ===" | tee "$outdir/runner.log"

    for attempt in 1 2; do
        if (( attempt == 1 )); then
            num_workers=$WEATHER_TEST_WORKERS
            persistent_workers=$WEATHER_TEST_PERSISTENT_WORKERS
            pin_memory=$WEATHER_TEST_PIN_MEMORY
            prefetch_factor=$WEATHER_TEST_PREFETCH_FACTOR
        else
            num_workers=0
            persistent_workers=False
            pin_memory=False
            prefetch_factor=2
            log "weather eval retry with minimal loader: epoch=${target_epoch} group=${group}"
        fi

        build_test_cfg_options "$ann_file" "$num_workers" "$persistent_workers" "$pin_memory" "$prefetch_factor"

        if (
            export PATH="$(dirname "$PYTHON_BIN"):$PATH"
            export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
            export PORT="$test_port"
            bash "$ROOT_DIR/tools/dist_test.sh" "$CONFIG" "$checkpoint" "$GPUS" \
                --work-dir "$outdir" \
                --cfg-options "${TEST_CFG_OPTIONS[@]}"
        ) > "$test_log" 2>&1; then
            break
        fi

        if (( attempt == 2 )); then
            log "weather eval failed after retries: epoch=${target_epoch} group=${group}"
            return 1
        fi
    done

    echo "=== DONE ${group} 2GPU ===" | tee -a "$outdir/runner.log"

    nds=$(extract_metric "$test_log" 'NDS')
    map=$(extract_metric "$test_log" 'mAP')
    log "epoch=${target_epoch} group=${group} NDS=${nds:-NA} mAP=${map:-NA}"
    record_summary "$target_epoch" "$group" "${nds:-NA}" "${map:-NA}"
}

ensure_epoch_weather_eval() {
    local target_epoch="$1"

    if (( target_epoch <= 0 )); then
        return 0
    fi

    if [[ ! -f "$WORK_DIR/epoch_${target_epoch}.pth" ]]; then
        log "skip weather eval for epoch ${target_epoch}: checkpoint missing"
        return 0
    fi

    for group in day night rain; do
        local group_index
        case "$group" in
            day) group_index=1 ;;
            night) group_index=2 ;;
            rain) group_index=3 ;;
        esac

        if weather_eval_completed "$target_epoch" "$group"; then
            log "weather eval already present: epoch=${target_epoch} group=${group}"
            continue
        fi

        log "backfilling missing weather eval: epoch=${target_epoch} group=${group}"
        run_weather_eval "$target_epoch" "$group" "$group_index"
    done
}

main() {
    local start_epoch target_epoch

    require_ann_splits

    if [[ ! -f "$SUMMARY_FILE" ]]; then
        printf 'epoch\tgroup\tNDS\tmAP\n' > "$SUMMARY_FILE"
    fi
    if [[ ! -f "$FULL_SUMMARY_FILE" ]]; then
        printf 'epoch\tNDS\tmAP\n' > "$FULL_SUMMARY_FILE"
    fi

    start_epoch=$(current_epoch)
    log "detected current epoch ${start_epoch} from last_checkpoint"

    ensure_epoch_weather_eval "$start_epoch"

    if (( start_epoch >= MAX_EPOCH )); then
        log "nothing to do: current epoch ${start_epoch} >= max epoch ${MAX_EPOCH}"
        return 0
    fi

    for ((target_epoch = start_epoch + 1; target_epoch <= MAX_EPOCH; target_epoch++)); do
        run_train_epoch "$target_epoch"
        run_weather_eval "$target_epoch" day 1
        run_weather_eval "$target_epoch" night 2
        run_weather_eval "$target_epoch" rain 3
    done

    log "completed epoch-wise training through epoch ${MAX_EPOCH}"
}

main "$@"