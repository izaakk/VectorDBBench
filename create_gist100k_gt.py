import polars as pl
import numpy as np

train_path = "/tmp/vectordb_bench/dataset/gist/gist_small_100k/train.parquet"
test_path = "/tmp/vectordb_bench/dataset/gist/gist_small_100k/test.parquet"
gt_path = "/tmp/vectordb_bench/dataset/gist/gist_small_100k/neighbors.parquet"

print("Loading train and test data...")
train_df = pl.read_parquet(train_path).sort("id")
test_df = pl.read_parquet(test_path).sort("id")

train_emb = np.stack(train_df["emb"])
train_ids = np.array(train_df["id"], dtype=np.int64)
test_emb = np.stack(test_df["emb"])
k = 100

neighbors_list = []

try:
    import faiss
    print("Using FAISS for fast ground truth computation...")
    index = faiss.IndexFlatL2(train_emb.shape[1])
    index.add(train_emb.astype(np.float32))
    D, I = index.search(test_emb.astype(np.float32), k)
    for idx in range(I.shape[0]):
        neighbors_list.append(train_ids[I[idx]].astype(int).tolist())
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} / {len(test_emb)} test vectors...")
except ImportError:
    print("FAISS not available, using brute-force NumPy fallback...")
    for idx, query in enumerate(test_emb):
        dists = np.linalg.norm(train_emb - query, axis=1)
        nn_idx = np.argpartition(dists, k)[:k]
        nn_idx = nn_idx[np.argsort(dists[nn_idx])]
        nn_ids = train_ids[nn_idx]
        neighbors_list.append(nn_ids.tolist())
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} / {len(test_emb)} test vectors...")

print("Saving ground truth file...")
ids = test_df["id"].cast(pl.Int64)
gt_df = pl.DataFrame({
    "id": ids,
    "neighbors_id": [list(map(int, n)) for n in neighbors_list]
})
gt_df.write_parquet(gt_path)
print(f"Saved ground truth to {gt_path}")
print(gt_df.head())
print(gt_df.schema)
