# Valkey Production Features Integration Plan

**Date:** 2026-04-01
**Branch:** valkey-production-features
**Base:** izaakk/VectorDBBench main branch

---

## Repository State Analysis

### izaakk/VectorDBBench (Current Base)

**Existing Features:**
- ✅ Basic Redis client (redis.py, config.py, cli.py)
- ✅ Redis HNSW support (RedisHNSWConfig)
- ❌ No SVS support
- ❌ No compression validation
- ❌ No calibration support
- ❌ No native logging

**Recent Commits:**
- SVS Vamana search_window_size support
- LeanVec model detection (but no SVS config)
- CohereIP dataset for Inner Product benchmarking

### mihaic/VectorDBBench (redis-filtering-calibration-svs)

**Features to Integrate:**
- ✅ RedisSVSVAMANAConfig class (complete SVS support)
- ✅ CLI command: redissvsvamana
- ✅ Calibration support (calibration_target, calibration_limit)
- ✅ Float16 support (use_float16)
- ✅ Filtering batch size (filtering_batch_size)
- ⚠️  SVS_VAMANA_COMPRESSION_OPTIONS = ["LeanVec4x8", "LVQ8"] ← Needs fix
- ⚠️  compression: Literal["LeanVec4x8", "LVQ8"] ← Needs fix

### Our Custom Fixes (from runtime patches)

**Fix 1: Batch Size Optimization**
- File: redis.py
- Change: batch_size from 20 → configurable via VECTORDB_BATCH_SIZE env (default 5)
- Reason: Prevents pipeline overload on Valkey

**Fix 2: Off-by-One Flush Fix**
- File: redis.py
- Change: `if i % batch_size == 0:` → `if (i + 1) % batch_size == 0:`
- Reason: Ensures final batch is flushed

**Fix 3a: SVS Index Type Normalization**
- File: redis.py
- Change: Normalize "SVS-VAMANA" → "SVS" for redis-py compatibility
- Reason: redis-py VectorField expects "SVS" algorithm name

**Fix 4: Compression Validation (3-Layer)**
- File: config.py
- Changes:
  - Line 4: SVS_VAMANA_COMPRESSION_OPTIONS = ["NONE", "FP16", "LVQ4", "LVQ8", "LVQ4X4", "LVQ4X8"]
  - Line ~75: compression: Literal["NONE", "FP16", "LVQ4", "LVQ8", "LVQ4X4", "LVQ4X8"] | None = None
  - index_param(): Add runtime validation and uppercase normalization
- Reason: Use only official SVS compression types from protobuf schema

**Fix 5: Native Command Logging**
- File: redis.py
- Change: Add logging before rs.create_index() to capture all FT.CREATE parameters
- Reason: Visibility into compression and other parameters without Redis MONITOR

---

## Integration Strategy

### Phase 1: Base SVS Support (from mihaic)
1. Copy RedisSVSVAMANAConfig class → config.py
2. Add redissvsvamana CLI command → cli.py
3. Keep calibration, float16, filtering features

### Phase 2: Apply Our Custom Fixes
1. Update SVS_VAMANA_COMPRESSION_OPTIONS to official types
2. Update Pydantic Literal annotation to official types
3. Add runtime compression validation in index_param()
4. Add native logging in make_index()
5. Add configurable batch_size (via env var)
6. Fix off-by-one flush bug
7. Add SVS index type normalization

### Phase 3: Testing & Validation
1. Local syntax check
2. Local development install test
3. Remote deployment to valkey-vector-1 and valkey-client-1
4. Validation test with Cohere-1M dataset
5. Verify compression via FT.INFO
6. Verify native logging captures parameters

---

## Official SVS Compression Types

From SVS protobuf schema (SVSCompressionType enum):
```protobuf
enum SVSCompressionType {
  SVS_COMPRESSION_NONE = 0;
  SVS_COMPRESSION_FP16 = 1;
  SVS_COMPRESSION_LVQ4 = 2;
  SVS_COMPRESSION_LVQ8 = 3;
  SVS_COMPRESSION_LVQ4X4 = 4;
  SVS_COMPRESSION_LVQ4X8 = 5;
}
```

**Parameter names:** NONE, FP16, LVQ4, LVQ8, LVQ4X4, LVQ4X8

**Note:** "LeanVec" is NOT an official SVS type!

---

## Files to Modify

1. `vectordb_bench/backend/clients/redis/config.py`
   - Add RedisSVSVAMANAConfig class
   - Update SVS_VAMANA_COMPRESSION_OPTIONS list
   - Update compression Literal annotation
   - Add runtime compression validation

2. `vectordb_bench/backend/clients/redis/cli.py`
   - Add redissvsvamana command
   - Add SVS-specific CLI parameters

3. `vectordb_bench/backend/clients/redis/redis.py`
   - Add native logging in make_index()
   - Add configurable batch_size
   - Fix off-by-one flush bug
   - Add SVS index type normalization

---

## Remotes Setup

- **origin:** izaakk/VectorDBBench (our working fork)
- **upstream:** zilliztech/VectorDBBench (official upstream)
- **mihaic:** mihaic/VectorDBBench (SVS features source)

---

## Success Criteria

After integration:
1. ✅ vectordbbench redissvsvamana command exists
2. ✅ Accepts official SVS compression types (--compression LVQ4X8)
3. ✅ Native logging captures FT.CREATE parameters
4. ✅ Index created with verified compression
5. ✅ All Python syntax valid
6. ✅ Local and remote tests pass
7. ✅ No runtime patching required in Ansible

---

## Next Steps

1. Task 2: Integrate compression validation fixes
2. Task 3: Integrate native logging
3. Task 4: Integrate batch size optimization
4. Task 5: Integrate off-by-one flush fix
5. Task 6: Integrate SVS index type normalization
6. Task 7: Test locally
7. Task 8: Push to GitHub
8. Task 9: Update Ansible config
9. Task 10: Deploy to remote servers
10. Task 11: Validation test
11. Task 12: Final documentation
