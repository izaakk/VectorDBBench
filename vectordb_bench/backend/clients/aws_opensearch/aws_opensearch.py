import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager

from opensearchpy import OpenSearch

from ..api import IndexType, VectorDB
from .config import AWSOpenSearchConfig, AWSOpenSearchIndexConfig, AWSOS_Engine

log = logging.getLogger(__name__)

WAITING_FOR_REFRESH_SEC = 30
WAITING_FOR_FORCE_MERGE_SEC = 30
SECONDS_WAITING_FOR_REPLICAS_TO_BE_ENABLED_SEC = 30


class AWSOpenSearch(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: AWSOpenSearchIndexConfig,
        index_name: str = "vdb_svs_fp16",  # must be lowercase
        id_col_name: str = "_id",
        vector_col_name: str = "target_field",
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        # HARDCODED for testing
        self.index_name = "target_index_hnsw"
        self.id_col_name = id_col_name
        self.category_col_names = [f"scalar-{categoryCount}" for categoryCount in [2, 5, 10, 100, 1000]]
        self.vector_col_name = vector_col_name

        log.info(f"AWS_OpenSearch client config: {self.db_config}")
        log.info(f"AWS_OpenSearch db case config : {self.case_config}")
        client = OpenSearch(**self.db_config)
        if drop_old:
            log.info(f"AWS_OpenSearch client drop old index: {self.index_name}")
            is_existed = client.indices.exists(index=self.index_name)
            if is_existed:
                client.indices.delete(index=self.index_name)
            self._create_index(client)

    @classmethod
    def config_cls(cls) -> AWSOpenSearchConfig:
        return AWSOpenSearchConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> AWSOpenSearchIndexConfig:
        return AWSOpenSearchIndexConfig

    def _create_index(self, client: OpenSearch):
        cluster_settings_body = {
            "persistent": {
                "knn.algo_param.index_thread_qty": self.case_config.index_thread_qty,
                "knn.memory.circuit_breaker.limit": self.case_config.cb_threshold,
            }
        }
        client.cluster.put_settings(body=cluster_settings_body)
        settings = {
            "index": {
                "knn": True,
                "number_of_shards": self.case_config.number_of_shards,
                "number_of_replicas": 0,
                "translog.flush_threshold_size": self.case_config.flush_threshold_size,
                "knn.advanced.approximate_threshold": "-1",
            },
            "refresh_interval": self.case_config.refresh_interval,
        }
        mappings = {
            "_source": {"excludes": [self.vector_col_name], "recovery_source_excludes": [self.vector_col_name]},
            "properties": {
                **{categoryCol: {"type": "keyword"} for categoryCol in self.category_col_names},
                self.vector_col_name: {
                    "type": "knn_vector",
                    "dimension": self.dim,
                    "method": self.case_config.index_param(),
                },
            },
        }
        try:
            client.indices.create(
                index=self.index_name,
                body={"settings": settings, "mappings": mappings},
            )
        except Exception as e:
            log.warning(f"Failed to create index: {self.index_name} error: {e!s}")
            raise e from None

    @contextmanager
    def init(self) -> None:
        """connect to opensearch"""
        self.client = OpenSearch(**self.db_config)
        yield
        self.client = None
        del self.client

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert the embeddings to the opensearch."""
        assert self.client is not None, "should self.init() first"

        insert_data = []
        for i in range(len(embeddings)):
            insert_data.append(
                {"index": {"_index": self.index_name, self.id_col_name: metadata[i]}},
            )
            insert_data.append({self.vector_col_name: embeddings[i]})
        try:
            resp = self.client.bulk(body=insert_data)
            log.info(f"AWS_OpenSearch adding documents: {len(resp['items'])}")
            resp = self.client.indices.stats(index=self.index_name)
            log.info(
                f"Total document count in index: {resp['_all']['primaries']['indexing']['index_total']}",
            )
            return (len(embeddings), None)
        except Exception as e:
            log.warning(f"Failed to insert data: {self.index_name} error: {e!s}")
            time.sleep(10)
            return self.insert_embeddings(embeddings, metadata)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        """Get k most similar embeddings to query vector."""
        assert self.client is not None, "should self.init() first"

        # Build KNN query
        knn_query = {
            "vector": query,
            "k": k,
        }

        # Handle method-specific runtime parameters safely
        svs_method = getattr(self.case_config, "svs_method", None)
        if svs_method is None:
            # Only apply ef_search for Lucene/HNSW, not for FAISS SVS
            if self.case_config.engine == AWSOS_Engine.lucene:
                knn_query["method_parameters"] = {"ef_search": self.case_config.efSearch}
            else:
                log.debug("Skipping ef_search for FAISS engine with no explicit method (likely SVS)")
        elif svs_method.lower() in ["svs_vamana", "vamana", "svs_flat", "flat"]:
            log.debug(f"Skipping ef_search for SVS method={svs_method}")
        else:
            knn_query["method_parameters"] = {"ef_search": self.case_config.efSearch}

        body = {
            "size": k,
            "query": {"knn": {self.vector_col_name: knn_query}},
            **({"filter": {"range": {self.id_col_name: {"gt": filters["id"]}}}} if filters else {}),
        }

        try:
            resp = self.client.search(
                index=self.index_name,
                body=body,
                size=k,
                _source=[self.id_col_name],
            )
            log.debug(f"Search took: {resp['took']}")
            log.debug(f"Search shards: {resp['_shards']}")
            log.debug(f"Search hits total: {resp['hits']['total']}")
            return [int(h["_id"]) for h in resp["hits"]["hits"]]
        except Exception as e:
            log.warning(f"Failed to search: {self.index_name} error: {e!s}")
            raise e from None

    def optimize(self, data_size: int | None = None):
        """Optimize index between insertion and search."""
        self._refresh_index()
        if self.case_config.force_merge_enabled:
            self._do_force_merge()
            self._refresh_index()
        self._update_replicas()
        self._refresh_index()
        self._load_graphs_to_memory()

    def _update_replicas(self):
        index_settings = self.client.indices.get_settings(index=self.index_name)
        current_number_of_replicas = int(index_settings[self.index_name]["settings"]["index"]["number_of_replicas"])
        log.info(
            f"Current Number of replicas are {current_number_of_replicas} "
            f"and changing to {self.case_config.number_of_replicas}"
        )
        settings_body = {"index": {"number_of_replicas": self.case_config.number_of_replicas}}
        self.client.indices.put_settings(index=self.index_name, body=settings_body)
        self._wait_till_green()

    def _wait_till_green(self):
        log.info("Wait for index to become green..")
        while True:
            res = self.client.cat.indices(index=self.index_name, h="health", format="json")
            health = res[0]["health"]
            if health == "green":
                break
            log.info(f"Index {self.index_name} has health: {health}. Retrying...")
            time.sleep(SECONDS_WAITING_FOR_REPLICAS_TO_BE_ENABLED_SEC)
        log.info(f"Index {self.index_name} is green.")

    def _refresh_index(self):
        log.debug(f"Starting refresh for index {self.index_name}")
        while True:
            try:
                log.info("Starting the Refresh Index..")
                self.client.indices.refresh(index=self.index_name)
                break
            except Exception as e:
                log.info(f"Refresh errored. Sleeping for {WAITING_FOR_REFRESH_SEC}s then retrying: {e}")
                time.sleep(WAITING_FOR_REFRESH_SEC)
                continue
        log.debug(f"Completed refresh for index {self.index_name}")

    def _do_force_merge(self):
        log.info(f"Updating index thread qty to {self.case_config.index_thread_qty_during_force_merge}.")
        cluster_settings_body = {
            "persistent": {"knn.algo_param.index_thread_qty": self.case_config.index_thread_qty_during_force_merge}
        }
        self.client.cluster.put_settings(body=cluster_settings_body)

        log.info("Updating graph threshold to allow merge-time graph creation.")
        output = self.client.indices.put_settings(
            index=self.index_name, body={"index.knn.advanced.approximate_threshold": "0"}
        )
        log.info(f"Response of updating setting: {output}")

        log.debug(f"Starting force merge for index {self.index_name}")
        segments = self.case_config.number_of_segments
        force_merge_endpoint = f"/{self.index_name}/_forcemerge?max_num_segments={segments}&wait_for_completion=false"
        force_merge_task_id = self.client.transport.perform_request("POST", force_merge_endpoint)["task"]
        while True:
            time.sleep(WAITING_FOR_FORCE_MERGE_SEC)
            task_status = self.client.tasks.get(task_id=force_merge_task_id)
            if task_status["completed"]:
                break
        log.debug(f"Completed force merge for index {self.index_name}")

    def _load_graphs_to_memory(self):
        if self.case_config.engine != AWSOS_Engine.lucene:
            log.info("Calling warmup API to load graphs into memory")
            warmup_endpoint = f"/_plugins/_knn/warmup/{self.index_name}"
            self.client.transport.perform_request("GET", warmup_endpoint)
