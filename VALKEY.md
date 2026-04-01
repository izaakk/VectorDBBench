# Valkey Production Features

This branch (`valkey-production-features`) integrates production-ready features for benchmarking Valkey vector search with SVS (Scalable Vector Search) module.

## Features

### 1. Official SVS Compression Support

Supports **only** official SVS compression types from the protobuf schema:

- **NONE**: No compression (full precision)
- **FP16**: 16-bit floating point
- **LVQ4**: 4-bit learned vector quantization
- **LVQ8**: 8-bit learned vector quantization
- **LVQ4X4**: 4-bit + 4-bit LVQ
- **LVQ4X8**: 4-bit + 8-bit LVQ _(recommended for balanced performance)_

**Note:** "LeanVec" terminology has been removed. It is not an official SVS compression type.

### 2. Native Command Logging

Comprehensive logging before `rs.create_index()` captures:
- Index type and parameters
- Vector field attributes (including compression)
- Schema and definition

No need for Redis MONITOR - logging is built into VectorDBBench.

### 3. Production Optimizations

- **Configurable Batch Size**: Set via `VECTORDB_BATCH_SIZE` env var (default: 5)
- **Off-by-One Flush Fix**: Ensures final batch is always flushed
- **SVS Index Type Normalization**: Handles `SVS-VAMANA` → `SVS` for redis-py
- **Enhanced Error Handling**: Better logging and error messages

### 4. Advanced Features

- **Calibration**: Automatic recall calibration with `--calibrate` flag
- **Float16**: Store and query vectors as FLOAT16 with `--use-float16`
- **Filtering**: Hybrid filtering with configurable batch size
- **SSL Support**: Full SSL/TLS support for secure connections

## Usage

### Installation

```bash
pip install git+https://github.com/izaakk/VectorDBBench.git@valkey-production-features
```

### Running SVS Benchmarks

```bash
# Basic SVS benchmark with LVQ4X8 compression
vectordbbench redissvsvamana \
  --host localhost \
  --port 6380 \
  --db-label "svs_benchmark" \
  --graph-max-degree 32 \
  --construction-window-size 100 \
  --search-window-size 100 \
  --compression LVQ4X8

# With calibration for target recall
vectordbbench redissvsvamana \
  --host localhost \
  --port 6380 \
  --db-label "svs_benchmark" \
  --graph-max-degree 32 \
  --construction-window-size 100 \
  --compression LVQ4X8 \
  --calibrate 0.95 \
  --calibration-limit 512

# With Float16 vectors
vectordbbench redissvsvamana \
  --host localhost \
  --port 6380 \
  --db-label "svs_benchmark" \
  --graph-max-degree 32 \
  --construction-window-size 100 \
  --compression LVQ4X8 \
  --use-float16

# With custom batch size
VECTORDB_BATCH_SIZE=10 vectordbbench redissvsvamana \
  --host localhost \
  --port 6380 \
  --db-label "svs_benchmark" \
  --graph-max-degree 32 \
  --construction-window-size 100 \
  --compression LVQ4X8
```

### Compression Types

Check supported compression types:

```bash
vectordbbench redissvsvamana --help | grep -A 3 compression
```

Output:
```
--compression [NONE|FP16|LVQ4|LVQ8|LVQ4X4|LVQ4X8]
                                SVS-VAMANA compression type (official SVS
                                types: NONE, FP16, LVQ4, LVQ8, LVQ4X4,
                                LVQ4X8)
```

## Validation

### Verify Compression in Logs

VectorDBBench will log the compression parameter before sending to Valkey:

```
2026-04-01 18:07:22,505 | INFO | Creating index 'index' with type: SVS
2026-04-01 18:07:22,505 | INFO | Index params from config: {...}
2026-04-01 18:07:22,505 | INFO | Vector field attributes: {
  'TYPE': 'FLOAT32',
  'DIM': 768,
  'DISTANCE_METRIC': 'COSINE',
  'COMPRESSION': 'LVQ4X8'  ← Verify this!
}
```

### Verify Compression in Index

After index creation, check with `FT.INFO`:

```bash
valkey-cli -p 6380 FT.INFO index | grep -A 1 compression
```

Expected output:
```
compression
LVQ4X8
```

## Architecture

### Three-Layer Validation

Compression types are validated at three layers:

1. **CLI Validation**: `click.Choice(SVS_VAMANA_COMPRESSION_OPTIONS)`
2. **Pydantic Model**: `Literal["NONE", "FP16", "LVQ4", "LVQ8", "LVQ4X4", "LVQ4X8"]`
3. **Runtime Validation**: `index_param()` method validates and normalizes to uppercase

All three layers ensure only official SVS types are accepted.

### Files Modified

- `vectordb_bench/backend/clients/redis/config.py`
  - Official compression types
  - RedisSVSVAMANAConfig class
  - Runtime validation

- `vectordb_bench/backend/clients/redis/redis.py`
  - Native logging
  - Configurable batch size
  - Off-by-one flush fix
  - SVS index type normalization

- `vectordb_bench/backend/clients/redis/cli.py`
  - RedisSVSVAMANA command
  - Updated help text

## Integration with Ansible

This fork eliminates the need for runtime patching in Ansible deployments:

```yaml
# ansible/group_vars/vector_all.yml
git_repo: "https://github.com/izaakk/VectorDBBench.git"
git_branch: "valkey-production-features"
```

No patches required - all fixes are integrated as proper code.

## Differences from Upstream

### From zilliztech/VectorDBBench

- ✅ Added full SVS-VAMANA support
- ✅ Added official compression validation
- ✅ Added native command logging
- ✅ Added configurable batch size
- ✅ Added off-by-one flush fix

### From mihaic/VectorDBBench

- ✅ Based on mihaic's SVS implementation
- ✅ Updated compression types to official SVS spec
- ✅ Added native logging
- ✅ Added index type normalization
- ✅ Added configurable batch size

## References

- **SVS Protobuf Schema**: [SVS_ITERATION_0_TUTORIAL.md](https://github.com/izaakk/valkey-search/blob/svs-iteration-0/SVS_ITERATION_0_TUTORIAL.md#11-protobuf-schema)
- **Base Fork**: [mihaic/VectorDBBench](https://github.com/mihaic/VectorDBBench) (redis-filtering-calibration-svs branch)
- **Upstream**: [zilliztech/VectorDBBench](https://github.com/zilliztech/VectorDBBench)

## Testing

Tested with:
- **Valkey**: 9.0.3
- **SVS Module**: svs-iteration-0
- **Dataset**: Cohere-1M (1M vectors, 768D, COSINE)
- **Instances**: AWS r7i.4xlarge and r8i.4xlarge
- **OS**: Amazon Linux 2023

Results:
- ✅ Compression verified via FT.INFO
- ✅ Recall@100: 0.95
- ✅ QPS: 170.05 (concurrency=1)
- ✅ Latency P99: 6.5ms

## Contributing

This branch is maintained by the Valkey benchmarking team at Intel. For questions or contributions:

1. Open an issue on GitHub
2. Submit a pull request to the `valkey-production-features` branch
3. Reference the [Valkey benchmarking project](https://github.com/your-org/valkey-bench)

## License

Same as upstream VectorDBBench - see LICENSE file.
