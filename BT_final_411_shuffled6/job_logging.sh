# Source after setting BUNDLE and VIEWS. Keep the Python exit status even when
# copying logs back to the shared filesystem fails.
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
LOCAL_LOG_DIR=$(mktemp -d "/var/tmp/BT-inner-${VIEWS}-${OAR_JOB_ID:-manual}.XXXXXX")
LOG_DEST="$BUNDLE/job_logs/${OAR_JOB_ID:-manual-$$}-$VIEWS"
exec 3>&1 4>&2
echo "Node: $(hostname). Local logs: $LOCAL_LOG_DIR. Final copy: $LOG_DEST" >&3
finish() {
    job_status=$?
    trap - EXIT TERM INT
    set +e
    printf '%s\n' "$job_status" > "$LOCAL_LOG_DIR/exit_status.txt"
    if mkdir -p "$LOG_DEST" && cp "$LOCAL_LOG_DIR/exit_status.txt" \
       "$LOCAL_LOG_DIR/train.out" "$LOCAL_LOG_DIR/train.err" "$LOG_DEST/"; then
        echo "Training exit code: $job_status. Full logs: $LOG_DEST" >&3
    else
        echo "Log copy failed. Exit code: $job_status. Local logs: $LOCAL_LOG_DIR" >&4
    fi
    exit "$job_status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > "$LOCAL_LOG_DIR/train.out" 2> "$LOCAL_LOG_DIR/train.err"
