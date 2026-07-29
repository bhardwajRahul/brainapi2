from src.adapters.embeddings import VectorStoreAdapter
from src.constants.embeddings import Vector


def vector_stable_id(vector: Vector) -> str:
    meta = vector.metadata or {}
    for key in ("uuid", "resource_id"):
        value = meta.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(vector.id or "")


def vector_rank_key(vector: Vector) -> tuple[float, str]:
    distance = (
        float("inf") if vector.distance is None else float(vector.distance)
    )
    return (distance, vector_stable_id(vector))


def stable_top_k_vectors(vectors: list[Vector], k: int) -> list[Vector]:
    if k <= 0:
        return []
    return sorted(vectors, key=vector_rank_key)[:k]


def ann_overfetch_k(k: int) -> int:
    if k <= 0:
        return 0
    return max(k * 4, k + 32)


class VectorSearchFacade:
    def __init__(self, vector_store: VectorStoreAdapter):
        self._vector_store = vector_store

    def search(
        self,
        data_vector: list[float],
        *,
        store: str,
        brain_id: str = "default",
        k: int = 10,
    ) -> list[Vector]:
        return self._vector_store.search_vectors(
            data_vector,
            store=store,
            brain_id=brain_id,
            k=k,
        )

    def search_nodes(
        self,
        data_vector: list[float],
        *,
        brain_id: str = "default",
        k: int = 10,
    ) -> list[Vector]:
        return self.search(data_vector, store="nodes", brain_id=brain_id, k=k)

    def search_triplets(
        self,
        data_vector: list[float],
        *,
        brain_id: str = "default",
        k: int = 10,
    ) -> list[Vector]:
        return self.search(data_vector, store="triplets", brain_id=brain_id, k=k)

    def search_relationships(
        self,
        data_vector: list[float],
        *,
        brain_id: str = "default",
        k: int = 10,
    ) -> list[Vector]:
        return self.search(data_vector, store="relationships", brain_id=brain_id, k=k)

    def search_data(
        self,
        data_vector: list[float],
        *,
        brain_id: str = "default",
        k: int = 10,
    ) -> list[Vector]:
        return self.search(data_vector, store="data", brain_id=brain_id, k=k)
