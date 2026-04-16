#!/bin/bash
# Full benchmark suite: HNSW vs SVS NONE vs SVS LVQ4X8
# Runs unattended — no user input required
BASEDIR="/home/ubuntu/projects/cee-valkey-svs"
cd "$BASEDIR"

LOGDIR="/tmp/benchmark_results"
mkdir -p "$LOGDIR"

SVS_LIB="$BASEDIR/valkey-search-svs/.build-release/_deps/svs-src/lib"
SERVER="$BASEDIR/valkey/src/valkey-server"
CLI="$BASEDIR/valkey/src/valkey-cli"
MODULE="$BASEDIR/valkey-search-svs/.build-release/libsearch.so"
LOADER="$BASEDIR/load_cohere_simple.py"
TUNER="$BASEDIR/tune_recall.py"

VDB_DIR="$BASEDIR/VectorDBBench"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOGDIR/run_all.log"; }

restart_server() {
    pkill valkey-server 2>/dev/null || true
    sleep 3
    rm -f dump.rdb
    LD_LIBRARY_PATH="$SVS_LIB:$LD_LIBRARY_PATH" "$SERVER" \
        --loadmodule "$MODULE" --port 6399 --loglevel notice --save "" \
        > "$LOGDIR/server.log" 2>&1 &
    sleep 3
    "$CLI" -p 6399 PING || { log "ERROR: Server failed to start"; exit 1; }
    log "Server started"
}

run_vectordbbench() {
    local algo="$1"
    local sws="$2"  # search_window_size for SVS or ef_runtime for HNSW
    local label="$3"

    log "Running VectorDBBench for $label (search param=$sws)..."

    if [ "$algo" = "hnsw" ]; then
        cd "$VDB_DIR"
        python3 -m vectordb_bench.cli.vectordbbench valkeysearchhnsw \
            --host localhost --port 6399 \
            --case-type Performance768D1M \
            --m 32 --ef-construction 128 --ef-runtime "$sws" \
            --skip-load \
            2>&1 | tee "$LOGDIR/vdbbench_${label}.log"
        cd "$BASEDIR"
    else
        cd "$VDB_DIR"
        python3 -m vectordb_bench.cli.vectordbbench valkeysearchsvs \
            --host localhost --port 6399 \
            --case-type Performance768D1M \
            --graph-max-degree 64 --construction-window-size 128 \
            --search-window-size "$sws" --alpha 1.0 \
            --skip-load \
            2>&1 | tee "$LOGDIR/vdbbench_${label}.log"
        cd "$BASEDIR"
    fi

    log "VectorDBBench complete for $label"
}

extract_results() {
    local label="$1"
    local logfile="$LOGDIR/vdbbench_${label}.log"

    log "=== Results: $label ==="
    grep -oP 'recall=np.float64\(\K[0-9.]+' "$logfile" | head -1 | xargs -I{} log "  Recall: {}"
    grep -oP 'ndcg=np.float64\(\K[0-9.]+' "$logfile" | head -1 | xargs -I{} log "  NDCG: {}"
    grep -oP 'qps=\K[0-9.]+' "$logfile" | head -1 | xargs -I{} log "  Peak QPS: {}"
    grep -oP 'serial_latency_p99=np.float64\(\K[0-9.]+' "$logfile" | head -1 | xargs -I{} log "  P99 Latency: {}"
    grep -oP 'serial_latency_p95=np.float64\(\K[0-9.]+' "$logfile" | head -1 | xargs -I{} log "  P95 Latency: {}"
}

########################################
# MAIN
########################################

log "========================================"
log "Starting full benchmark suite"
log "========================================"

# ---- 1. HNSW M=32, EF_CONSTRUCTION=128, COSINE ----
log ""
log "==== BENCHMARK 1: HNSW M=32 ===="
restart_server
log "Loading HNSW M=32 COSINE..."
python3 "$LOADER" --algorithm hnsw --flush-db 2>&1 | tee "$LOGDIR/load_hnsw.log"
log "Tuning HNSW recall..."
python3 "$TUNER" --algorithm hnsw --num-queries 200 2>&1 | tee "$LOGDIR/tune_hnsw.log"

# Find EF_RUNTIME closest to 95% recall from tune output
HNSW_EF=$(python3 -c "
import re, sys
best_ef, best_diff = 100, 1.0
for line in open('$LOGDIR/tune_hnsw.log'):
    m = re.search(r'(\d+)\s+([\d.]+)\s+[\d.]+ms', line)
    if m:
        ef, recall = int(m.group(1)), float(m.group(2))
        diff = abs(recall - 0.95)
        if diff < best_diff:
            best_diff = diff
            best_ef = ef
print(best_ef)
")
log "Selected EF_RUNTIME=$HNSW_EF for HNSW benchmark"
run_vectordbbench hnsw "$HNSW_EF" "hnsw_m32"
extract_results "hnsw_m32"

# ---- 2. SVS NONE, GRAPH_MAX_DEGREE=64, COSINE ----
log ""
log "==== BENCHMARK 2: SVS NONE (GMD=64) ===="
restart_server
log "Loading SVS NONE COSINE..."
python3 "$LOADER" --algorithm svs --flush-db 2>&1 | tee "$LOGDIR/load_svs_none.log"
log "Tuning SVS NONE recall..."
python3 "$TUNER" --algorithm svs --num-queries 200 2>&1 | tee "$LOGDIR/tune_svs_none.log"

SVS_SWS=$(python3 -c "
import re, sys
best_sws, best_diff = 200, 1.0
for line in open('$LOGDIR/tune_svs_none.log'):
    m = re.search(r'(\d+)\s+([\d.]+)\s+[\d.]+ms', line)
    if m:
        sws, recall = int(m.group(1)), float(m.group(2))
        diff = abs(recall - 0.95)
        if diff < best_diff:
            best_diff = diff
            best_sws = sws
print(best_sws)
")
log "Selected SEARCH_WINDOW_SIZE=$SVS_SWS for SVS NONE benchmark"
run_vectordbbench svs "$SVS_SWS" "svs_none"
extract_results "svs_none"

# ---- 3. SVS LVQ4X8, GRAPH_MAX_DEGREE=64, COSINE ----
log ""
log "==== BENCHMARK 3: SVS LVQ4X8 (GMD=64) ===="
restart_server
log "Loading SVS LVQ4X8 COSINE..."
python3 "$LOADER" --algorithm svs --compression LVQ4X8 --flush-db 2>&1 | tee "$LOGDIR/load_svs_lvq4x8.log"
log "Tuning SVS LVQ4X8 recall..."
python3 "$TUNER" --algorithm svs --num-queries 200 2>&1 | tee "$LOGDIR/tune_svs_lvq4x8.log"

SVS_LVQ_SWS=$(python3 -c "
import re, sys
best_sws, best_diff = 200, 1.0
for line in open('$LOGDIR/tune_svs_lvq4x8.log'):
    m = re.search(r'(\d+)\s+([\d.]+)\s+[\d.]+ms', line)
    if m:
        sws, recall = int(m.group(1)), float(m.group(2))
        diff = abs(recall - 0.95)
        if diff < best_diff:
            best_diff = diff
            best_sws = sws
print(best_sws)
")
log "Selected SEARCH_WINDOW_SIZE=$SVS_LVQ_SWS for SVS LVQ4X8 benchmark"
run_vectordbbench svs "$SVS_LVQ_SWS" "svs_lvq4x8"
extract_results "svs_lvq4x8"

# ---- Summary ----
log ""
log "========================================"
log "ALL BENCHMARKS COMPLETE"
log "========================================"
log ""
log "Results summary:"
log "  HNSW M=32:    see $LOGDIR/vdbbench_hnsw_m32.log"
log "  SVS NONE:     see $LOGDIR/vdbbench_svs_none.log"
log "  SVS LVQ4X8:   see $LOGDIR/vdbbench_svs_lvq4x8.log"
log "  Full log:     $LOGDIR/run_all.log"

# Kill server at the end
pkill valkey-server 2>/dev/null || true
log "Server stopped. Done."
