import logging
import os
from contextlib import contextmanager
from typing import Any

import numpy as np
import redis
from redis.commands.search.query import Query

from vectordb_bench.backend.filter import Filter, FilterOp
from ..api import DBCaseConfig, VectorDB

log = logging.getLogger(__name__)
INDEX_NAME = "index"  # Vector Index Name


class Redis(VectorDB):

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = INDEX_NAME
        self.with_scalar_labels = with_scalar_labels
        self._filter = "*"
        self._vector_field = "vector"
        self._label_field = "label"
        self._numeric_field = "metadata"
        self._np_dtype = np.float16 if db_case_config.use_float16 else np.float32
        self._redis_type = "FLOAT16" if db_case_config.use_float16 else "FLOAT32"

        # Create a redis connection, if db has password configured, add it to the connection here and in init():
        password = self.db_config["password"]

        # Get connection parameters with proper defaults
        conn_params = {
            "host": self.db_config["host"],
            "port": self.db_config["port"],
            "password": password,
            "db": 0,
        }

        # Add SSL parameters if configured
        if self.db_config.get("ssl", False):
            conn_params["ssl"] = True
            if ssl_ca_certs := self.db_config.get("ssl_ca_certs"):
                conn_params["ssl_ca_certs"] = ssl_ca_certs

        # Handle CMD (Cluster Mode Disabled) connection
        if self.db_config.get("cmd", False):
            conn = redis.Redis(**conn_params)
        else:
            conn = redis.Redis(**conn_params)

        if drop_old:
            try:
                conn.ft(INDEX_NAME).info()
                conn.ft(INDEX_NAME).dropindex()
                log.info(f"Redis client drop_old collection: {self.collection_name}")
            except redis.exceptions.ResponseError:
                log.info(f"Redis client no existing index to drop: {self.collection_name}")

        self.make_index(dim, conn)
        conn.close()
        conn = None

    def make_index(self, vector_dimensions: int, conn: redis.Redis):
        """Create index using raw execute_command for full parameter control.

        This bypasses redis-py's VectorField abstraction to ensure ALL parameters
        (including COMPRESSION) are passed correctly to the FT.CREATE command.
        """
        try:
            # Check to see if index exists
            conn.ft(INDEX_NAME).info()
            log.info(f"Index {INDEX_NAME} already exists, skipping creation")
            return
        except Exception:
            pass

        index_params = self.case_config.index_param()
        index_type = index_params["index_type"]
        params = index_params["params"]
        metric_type = index_params.get("metric_type", "COSINE")

        # Normalize index type: "SVS-VAMANA" → "SVS"
        if index_type == "SVS-VAMANA":
            log.info(f"Normalizing algorithm: '{index_type}' → 'SVS'")
            algorithm = "SVS"
        elif index_type in ["HNSW", "FLAT", "SVS"]:
            algorithm = index_type
        else:
            log.warning(f"Unknown index type '{index_type}', defaulting to 'FLAT'")
            algorithm = "FLAT"

        # Build vector parameters as flat key-value list
        vector_params = [
            "TYPE", self._redis_type,
            "DIM", str(vector_dimensions),
            "DISTANCE_METRIC", metric_type,
        ]

        # Add algorithm-specific parameters from config
        for key, value in params.items():
            # Skip hybrid_policy and filtering_batch_size (runtime params, not index params)
            if key in ["hybrid_policy", "filtering_batch_size"]:
                continue
            vector_params.extend([str(key), str(value)])

        log.info(f"Creating index '{INDEX_NAME}' with algorithm: {algorithm}")
        log.info(f"Vector parameters: {vector_params}")

        # Build FT.CREATE command
        cmd = [
            "FT.CREATE", INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", f"{INDEX_NAME}:",
            "SCHEMA",
        ]

        # Add numeric field for metadata filtering
        cmd.extend([self._numeric_field, "NUMERIC"])

        # Add label field if enabled
        if self.with_scalar_labels:
            cmd.extend([self._label_field, "TAG"])

        # Add vector field with all parameters
        cmd.extend([
            self._vector_field, "VECTOR", algorithm, str(len(vector_params))
        ])
        cmd.extend(vector_params)

        log.info(f"FT.CREATE command: {' '.join(cmd)}")

        # Execute raw command
        conn.execute_command(*cmd)
        log.info(f"Index '{INDEX_NAME}' created successfully with {algorithm} algorithm")

    @contextmanager
    def init(self) -> None:
        """create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        conn_params = {
            "host": self.db_config["host"],
            "port": self.db_config["port"],
            "password": self.db_config["password"],
            "db": 0,
        }

        if self.db_config.get("ssl", False):
            conn_params["ssl"] = True
            if ssl_ca_certs := self.db_config.get("ssl_ca_certs"):
                conn_params["ssl_ca_certs"] = ssl_ca_certs

        self.conn = redis.Redis(**conn_params)
        yield
        self.conn.close()
        self.conn = None

    def optimize(self, data_size: int | None = None):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        """Insert embeddings into the database.
        Should call self.init() first.
        """

        # Fix 1: Configurable batch size via environment variable
        # Default to 5 for Valkey (optimal based on testing), allow override
        batch_size = int(os.environ.get('VECTORDB_BATCH_SIZE', '5'))

        try:
            with self.conn.pipeline(transaction=False) as pipe:
                for i, embedding in enumerate(embeddings):
                    ndarr_emb = np.array(embedding).astype(self._np_dtype).tobytes()
                    mapping = {
                        self._vector_field: ndarr_emb,
                        self._numeric_field: metadata[i],
                    }
                    if self.with_scalar_labels:
                        assert labels_data is not None
                        mapping[self._label_field] = labels_data[i]
                    pipe.hset(
                        metadata[i],
                        mapping=mapping,
                    )
                    # Fix 2: Off-by-one flush fix to ensure final batch is flushed
                    # Changed from: if i % batch_size == 0:
                    # To: if (i + 1) % batch_size == 0:
                    if (i + 1) % batch_size == 0:
                        pipe.execute()

                # Final flush to catch any remaining items
                pipe.execute()
                result_len = i + 1
        except redis.exceptions.RedisError as e:
            log.error(f"Redis error during insert_embeddings: {e}")
            return 0, e
        except Exception as e:
            log.error(f"Unexpected error during insert_embeddings: {e}")
            return 0, e

        return result_len, None

    def prepare_filter(self, filters: Filter):
        # Handle None filters (can occur during subprocess serialization)
        if filters is None:
            self._filter = "*"
            return

        if filters.type == FilterOp.NonFilter:
            self._filter = "*"
        elif filters.type == FilterOp.NumGE:
            self._filter = f"@{self._numeric_field}:[{filters.int_value} +inf]"
        elif filters.type == FilterOp.StrEqual:
            self._filter = f"@{self._label_field}:{{ {filters.label_value} }}"
        else:
            msg = f"Not support Filter for Redis - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        config_overwrite: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None

        query_vector = np.array(query).astype(self._np_dtype).tobytes()
        search_params = self.case_config.search_param()["params"]
        is_filtering = self._filter != "*"

        if is_filtering:
            if (hybrid_policy := search_params['hybrid_policy']) == "BATCHES":
                if config_overwrite is not None and "filtering_batch_size" in config_overwrite:
                    filtering_batch_size = config_overwrite["filtering_batch_size"]
                else:
                    filtering_batch_size = search_params.get("filtering_batch_size")
                filtering_batch_size_params = f" BATCH_SIZE {filtering_batch_size}" if filtering_batch_size is not None else ""
            else:
                filtering_batch_size_params = ""
            filtering_params = f" HYBRID_POLICY {hybrid_policy}{filtering_batch_size_params}"
        else:
            filtering_params = ""

        runtime_param = self.case_config.knn_runtime_param(config_overwrite)
        query_obj = (
            Query(f"{self._filter}=>[KNN {k} @{self._vector_field} $vec {runtime_param}{filtering_params}]")
            .paging(0, k)
        )
        query_params = {"vec": query_vector}
        res = self.conn.ft(INDEX_NAME).search(query_obj, query_params)
        return [int(doc["id"]) for doc in res.docs]
