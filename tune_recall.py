#!/usr/bin/env python3
"""
Quick recall tuning: test different search parameters against ground truth.

Loads the VectorDBBench test/neighbors parquet files and runs KNN search
with varying EF_RUNTIME (HNSW) or SEARCH_WINDOW_SIZE (SVS) to find
the parameter that achieves ~95% recall.
"""
import argparse
import logging
import time

import numpy as np
import polars as pl
import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

INDEX_NAME = "vdbbench"
DATASET_DIR = "/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m"


def compute_recall(results: list[int], ground_truth: list[int], k: int) -> float:
    gt_set = set(ground_truth[:k])
    return len(set(results[:k]) & gt_set) / k


def run_search(conn: redis.Redis, query_vec: bytes, k: int, algo: str, param_value: int) -> list[int]:
    if algo == "hnsw":
        knn_extra = f" EF_RUNTIME {param_value}"
    else:
        knn_extra = f" SEARCH_WINDOW_SIZE {param_value}"

    query_str = f"*=>[KNN {k} @vector $vec{knn_extra}]"
    result = conn.execute_command(
        'FT.SEARCH', INDEX_NAME, query_str,
        'PARAMS', '2', 'vec', query_vec,
        'RETURN', '0',
        'LIMIT', '0', str(k),
        'DIALECT', '2',
    )
    count = result[0]
    ids = []
    for i in range(1, len(result)):
        key = result[i]
        if isinstance(key, bytes):
            key = key.decode()
        doc_id = int(key.split(":")[-1])
        ids.append(doc_id)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Tune search parameters for target recall")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6399)
    parser.add_argument("--algorithm", choices=["hnsw", "svs"], required=True)
    parser.add_argument("--num-queries", type=int, default=100, help="Number of test queries")
    parser.add_argument("--k", type=int, default=100, help="Top-K for recall")

    args = parser.parse_args()

    conn = redis.Redis(host=args.host, port=args.port, db=0)

    # Load test data and ground truth
    log.info("Loading test queries and ground truth...")
    test_df = pl.read_parquet(f"{DATASET_DIR}/test.parquet")
    gt_df = pl.read_parquet(f"{DATASET_DIR}/neighbors.parquet")

    test_vectors = test_df["emb"].to_list()
    gt_neighbors = gt_df["neighbors_id"].to_list()

    num_queries = min(args.num_queries, len(test_vectors))
    log.info(f"Using {num_queries} test queries, k={args.k}")

    # Parameter sweep — focused around 95% recall target
    if args.algorithm == "hnsw":
        param_name = "EF_RUNTIME"
        values = [150, 160, 170, 180, 190, 200, 210, 220]
    else:
        param_name = "SEARCH_WINDOW_SIZE"
        values = [100, 130, 150, 170, 190, 200, 220, 250]

    log.info(f"\n{'':=<70}")
    log.info(f"{'':>5}{param_name:>20}  {'Recall@100':>12}  {'Avg Latency':>12}  {'P99 Latency':>12}")
    log.info(f"{'':=<70}")

    for val in values:
        recalls = []
        latencies = []

        for i in range(num_queries):
            query_bytes = np.array(test_vectors[i], dtype=np.float32).tobytes()
            gt = gt_neighbors[i]

            t0 = time.time()
            results = run_search(conn, query_bytes, args.k, args.algorithm, val)
            latency = time.time() - t0
            latencies.append(latency)

            recall = compute_recall(results, gt, args.k)
            recalls.append(recall)

        avg_recall = np.mean(recalls)
        avg_latency = np.mean(latencies) * 1000
        p99_latency = np.percentile(latencies, 99) * 1000

        marker = " <-- target" if 0.945 <= avg_recall <= 0.965 else ""
        log.info(f"{'':>5}{val:>20}  {avg_recall:>12.4f}  {avg_latency:>10.1f}ms  {p99_latency:>10.1f}ms{marker}")

    log.info(f"{'':=<70}")


if __name__ == "__main__":
    main()
