from typing import Literal

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

# Official SVS compression types from protobuf schema (SVSCompressionType enum)
# See: https://github.com/izaakk/valkey-search/blob/svs-iteration-0/SVS_ITERATION_0_TUTORIAL.md#11-protobuf-schema
SVS_VAMANA_COMPRESSION_OPTIONS = ["NONE", "FP16", "LVQ4", "LVQ8", "LVQ4X4", "LVQ4X8"]


class RedisConfig(DBConfig):
    password: SecretStr | None = None
    host: SecretStr
    port: int | None = None
    ssl: bool = True
    ssl_ca_certs: str | None = None
    cmd: bool = False

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "password": self.password.get_secret_value() if self.password is not None else None,
            "ssl": self.ssl,
            "ssl_ca_certs": self.ssl_ca_certs,
            "decode_responses": False,
        }


class RedisIndexConfig(BaseModel):
    """Base config for Redis vector indexes"""

    metric_type: MetricType | None = None
    use_float16: bool = False
    filtering_batch_size: int | None = None
    calibration_target: float | None = None
    calibration_limit: int = 1000
    hybrid_policy: Literal["ADHOC_BF", "BATCHES"] = "BATCHES"

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""
        return self.metric_type.value


class RedisHNSWConfig(RedisIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    index: IndexType = IndexType.HNSW
    calibration_param: Literal["ef", "filtering_batch_size"] = "ef"

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "EF_CONSTRUCTION": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {
                "ef": self.ef,
                "calibration_target": self.calibration_target,
                "calibration_param": self.calibration_param,
                "calibration_limit": self.calibration_limit,
                "filtering_batch_size": self.filtering_batch_size,
                "hybrid_policy": self.hybrid_policy,
            },
        }

    def knn_runtime_param(self, config_overwrite: dict | None = None) -> str:
        ef = config_overwrite["ef"] if config_overwrite is not None and "ef" in config_overwrite else self.ef
        return f"EF_RUNTIME {ef}"


class RedisSVSVAMANAConfig(RedisIndexConfig, DBCaseConfig):
    """
    Configuration for Redis SVS-VAMANA index.

    SVS (Scalable Vector Search) with VAMANA graph algorithm.
    Supports official compression types from SVS protobuf schema.
    """
    graph_max_degree: int
    construction_window_size: int
    search_window_size: int | None = None
    # Official SVS compression types: NONE, FP16, LVQ4, LVQ8, LVQ4X4, LVQ4X8
    compression: Literal["NONE", "FP16", "LVQ4", "LVQ8", "LVQ4X4", "LVQ4X8"] | None = None
    index: IndexType = IndexType.SVS_VAMANA
    calibration_param: Literal["search_window_size", "filtering_batch_size"] = "search_window_size"

    def index_param(self) -> dict:
        """
        Generate FT.CREATE index parameters for SVS-VAMANA.

        Validates and normalizes compression type to uppercase.
        Only accepts official SVS compression types from protobuf schema.
        """
        params: dict = {
            "GRAPH_MAX_DEGREE": self.graph_max_degree,
            "CONSTRUCTION_WINDOW_SIZE": self.construction_window_size,
        }

        # Runtime validation and normalization of compression type
        if self.compression is not None:
            # Normalize to uppercase (protobuf enum convention)
            compression_upper = self.compression.upper()

            # Validate against official SVS types
            valid_svs_compression = ["NONE", "FP16", "LVQ4", "LVQ8", "LVQ4X4", "LVQ4X8"]
            if compression_upper not in valid_svs_compression:
                raise ValueError(
                    f"Invalid SVS compression type: {self.compression}. "
                    f"Must be one of {valid_svs_compression}. "
                    f"Note: 'LeanVec' is not an official SVS compression type."
                )

            params["COMPRESSION"] = compression_upper

        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": params,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {
                "search_window_size": self.search_window_size,
                "calibration_target": self.calibration_target,
                "calibration_param": self.calibration_param,
                "calibration_limit": self.calibration_limit,
                "filtering_batch_size": self.filtering_batch_size,
                "hybrid_policy": self.hybrid_policy,
            },
        }

    def knn_runtime_param(self, config_overwrite: dict | None = None) -> str:
        sws = config_overwrite["search_window_size"] if config_overwrite is not None and "search_window_size" in config_overwrite else self.search_window_size
        return f"SEARCH_WINDOW_SIZE {sws}"
