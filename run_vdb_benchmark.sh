#!/bin/bash
# Script to run VectorDBBench for a specific index
# Usage: ./run_vdb_benchmark.sh <index_name> <index_type> [metric] [concurrency_levels]
# Example: ./run_vdb_benchmark.sh cohere-1m-target svs_leanvec4x8 ip "50,75"

set -e

INDEX_NAME=$1
INDEX_TYPE=$2
METRIC=${3:-"ip"}  # Default to inner product
CONCURRENCY=${4:-"1,5,10,20,50,75,100"}  # Default to all levels

if [ -z "$INDEX_NAME" ] || [ -z "$INDEX_TYPE" ]; then
    echo "Usage: $0 <index_name> <index_type> [metric] [concurrency_levels]"
    echo "Example: $0 cohere-1m-target svs_leanvec4x8 ip \"50,75\""
    echo "Metric options: ip (default), l2, cosine"
    echo "Concurrency: comma-separated list (default: \"1,5,10,20,50,75,100\")"
    exit 1
fi

# OpenSearch cluster details
OS_HOST="10.0.0.78"
OS_PORT="9200"

# Determine case type based on metric
case "$METRIC" in
    "ip")
        CASE_TYPE="Performance768D1MIP"
        METRIC_LABEL="Inner Product"
        ;;
    "l2")
        CASE_TYPE="Performance768D1ML2"
        METRIC_LABEL="L2"
        ;;
    "cosine")
        CASE_TYPE="Performance768D1M"
        METRIC_LABEL="COSINE"
        ;;
    *)
        echo "Error: Invalid metric '$METRIC'. Use: ip, l2, or cosine"
        exit 1
        ;;
esac

echo "========================================="
echo "Running VectorDBBench for: $INDEX_NAME"
echo "Index type: $INDEX_TYPE"
echo "Dataset: Cohere-1M (${METRIC_LABEL} metric)"
echo "Case type: $CASE_TYPE"
echo "Concurrency levels: $CONCURRENCY"
echo "========================================="

# Run VectorDBBench with appropriate dataset
cd /home/ubuntu/VectorDBBench
 
python3.11 -m vectordb_bench.cli.vectordbbench awsopensearch\
  --db-label "opensearch-${INDEX_TYPE}-${METRIC}" \
  --host "$OS_HOST" \
  --port "$OS_PORT" \
  --case-type "$CASE_TYPE" \
  --engine "faiss" \
  --m 16 \
  --ef-construction 100 \
  --ef-runtime 256 \
  --skip-load \
  --data-source "Local" \
  --num-concurrency "$CONCURRENCY" \
  --concurrency-duration 60 \
  --task-label "svs-benchmark-${INDEX_TYPE}-$(date +%Y%m%d-%H%M)" \
  --index-name "$INDEX_NAME"

echo ""
echo "========================================="
echo "Benchmark completed for: $INDEX_NAME"
echo "Results saved in: /home/ubuntu/VectorDBBench/vectordb_bench/results/OpenSearch/"
echo "========================================="
