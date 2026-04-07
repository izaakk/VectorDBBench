import logging
import os
from contextlib import contextmanager
from typing import Any

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, VectorField
try:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
except ImportError:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
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
        try:
            # check to see if index exists
            conn.ft(INDEX_NAME).info()
            log.info(f"Index {INDEX_NAME} already exists, skipping creation")
        except Exception:
            index_params = self.case_config.index_param()
            index_type = index_params["index_type"]

            # redis-py is patched to accept "SVS-VAMANA" - pass through original value
            # Valkey expects the full "SVS-VAMANA" string in FT.CREATE command

            vector_field_attrs = {
                "TYPE": self._redis_type,  # FLOAT16, FLOAT32 or FLOAT64
                "DIM": vector_dimensions,  # Number of Vector Dimensions
                "DISTANCE_METRIC": "COSINE",  # Vector Search Distance Metric
                **index_params["params"],
            }

            # Create VectorField with original index_type for redis-py validation
            vector_field = VectorField(self._vector_field, index_type, vector_field_attrs)

            # Normalize for Valkey server compatibility
            # redis-py validates 'SVS-VAMANA' but Valkey expects just 'SVS'
            if index_type == "SVS-VAMANA":
                log.info(f"Normalizing algorithm for Valkey: '{index_type}' → 'SVS'")
                vector_field.args[1] = "SVS"  # args = [VECTOR, algorithm, count, ...]

            schema = [
                NumericField(self._numeric_field),
                vector_field,
            ]
            if self.with_scalar_labels:
                schema.append(TagField(self._label_field))

            definition = IndexDefinition(index_type=IndexType.HASH)

            rs = conn.ft(INDEX_NAME)

            # Fix 5: Native command logging for debugging and verification
            log.info(f"Creating index '{INDEX_NAME}' with type: {index_type}")
            log.info(f"Index params from config: {index_params}")
            log.info(f"Vector field attributes: {vector_field_attrs}")
            log.info(f"Schema: {schema}")
            log.info(f"Definition: {definition}")

            rs.create_index(schema, definition=definition)
            log.info(f"Index '{INDEX_NAME}' created successfully")

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
