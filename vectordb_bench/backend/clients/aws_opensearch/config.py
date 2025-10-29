import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType

log = logging.getLogger(__name__)


class AWSOpenSearchConfig(DBConfig, BaseModel):
    host: str = ""
    port: int = 80
    user: Optional[str] = None
    password: Optional[SecretStr] = None

    def to_dict(self) -> dict:
        use_ssl = self.port == 443 and self.host != "localhost"
        http_auth = (
            (self.user, self.password.get_secret_value())
            if self.user and self.password
            else ()
        )
        return {
            "hosts": [{"host": self.host, "port": self.port}],
            "http_auth": http_auth,
            "use_ssl": use_ssl,
            "http_compress": True,
            "verify_certs": use_ssl,
            "ssl_assert_hostname": False,
            "ssl_show_warn": False,
            "timeout": 600,
        }


class AWSOS_Engine(Enum):
    faiss = "faiss"
    lucene = "lucene"



# SVS encoder types
class AWSOSEncoder(Enum):
    none = "none"         # No encoder (flat or vamana)
    fp16 = "svs_fp16"
    sq8 = "svs_sq8"
    lvq = "lvq"           # Note: LVQ encoder name is 'lvq', not 'svs_lvq'

class AWSOSQuantization(Enum):
    fp32 = "fp32"
    fp16 = "fp16"



class AWSOpenSearchIndexConfig(BaseModel, DBCaseConfig):
    # General
    metric_type: MetricType = MetricType.L2
    engine: AWSOS_Engine = AWSOS_Engine.faiss
    # SVS
    svs_method: str | None = None  # flat, vamana
    svs_encoder: AWSOSEncoder = AWSOSEncoder.none  # none, fp16, sq8, lvq
    degree: int = 64  # Only used for Vamana
    primary_bits: int = 4  # For LVQ
    residual_bits: int = 8  # For LVQ
    # HNSW
    efConstruction: int = 256
    efSearch: int = 256
    M: int = 16
    # Other
    index_thread_qty: int | None = 4
    number_of_shards: int | None = 1
    number_of_replicas: int | None = 0
    number_of_segments: int | None = 1
    refresh_interval: str | None = "60s"
    force_merge_enabled: bool | None = True
    flush_threshold_size: str | None = "5120mb"
    index_thread_qty_during_force_merge: int = 4
    cb_threshold: str | None = "50%"
    quantization_type: AWSOSQuantization = AWSOSQuantization.fp32

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.IP:
            return "innerproduct"
        if self.metric_type == MetricType.COSINE:
            return "cosinesimil"
        return "l2"

    def index_param(self) -> dict:
        # Only use SVS if explicitly requested
        if self.svs_method == "flat":
            return {
                "name": "svs_flat",
                "space_type": self.parse_metric(),
                "engine": self.engine.value,
            }
        if self.svs_method == "vamana":
            params = {"degree": self.degree}
            if self.svs_encoder == AWSOSEncoder.fp16:
                params["encoder"] = {"name": "svs_fp16"}
            elif self.svs_encoder == AWSOSEncoder.sq8:
                params["encoder"] = {"name": "svs_sq8"}
            elif self.svs_encoder == AWSOSEncoder.lvq:
                params["encoder"] = {
                    "name": "lvq",
                    "parameters": {
                        "primary_bits": self.primary_bits,
                        "residual_bits": self.residual_bits,
                    },
                }
            return {
                "name": "svs_vamana",
                "space_type": self.parse_metric(),
                "engine": self.engine.value,
                "parameters": params,
            }
        # Default: HNSW (with or without quantization)
        params = {
            "ef_construction": self.efConstruction,
            "m": self.M,
            "ef_search": self.efSearch,
        }
        if self.quantization_type is not AWSOSQuantization.fp32:
            params["encoder"] = {"name": "sq", "parameters": {"type": self.quantization_type.fp16.value}}
        return {
            "name": "hnsw",
            "space_type": self.parse_metric(),
            "engine": self.engine.value,
            "parameters": params,
        }

    def search_param(self) -> dict:
        return {}
