from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models

from logger import Logger


class QdrantDBClient:
    def __init__(self, QDRANT_HOST):
        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST, port=6333, grpc_port=6334, prefer_grpc=True
        )
        self.logger = Logger().get_logger()

    def save_batch_embeddings(
        self, collection_name, embeddings, properties: List[dict]
    ):
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=[prop["external_id"] for prop in properties],
                vectors=embeddings,
                payloads=properties,
            ),
        )
        self.logger.info(f"Saved batch of embeddings {len(embeddings)} items")

    def save_embedding(self, collection_name, embedding, properties: dict):
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=properties["external_id"],
                    vector=embedding,
                    payload=properties,
                )
            ],
        )
        self.logger.info(f"Saved embedding {properties['external_id']}")

    def search(self, collection_name, query_vector, top_k):
        response = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return response

    def drop_collection(self, collection_name):
        self.logger.info(f"Deleting collection {collection_name}")
        self.qdrant_client.delete_collection(collection_name=collection_name)

    def create_collection(self, collection_name, vector_size, delete_if_exists=False):
        if self.qdrant_client.collection_exists(collection_name):
            self.logger.info(f"Collection {collection_name} already exists")
            if delete_if_exists:
                self.logger.info(f"Deleting collection {collection_name}")
                self.qdrant_client.delete_collection(collection_name)
            else:
                self.logger.info(f"Skipping collection {collection_name}")
                return

        self.logger.info(f"Creating collection {collection_name}")
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )

    def fetch_all(self, collection_name, include_vector: bool = False):
        result = self.qdrant_client.scroll(
            collection_name=collection_name,
            with_vectors=include_vector,
            with_payload=True,
        )
        for item in result:
            print(item)

    def get_db_size(self, collection_names=None):
        if collection_names is None:
            collection_names = self.qdrant_client.get_collections().collections

        for collection_name in collection_names:
            count = self.qdrant_client.count(collection_name).count
            print(f"{collection_name} size: {count}")

    def is_present_by_id(self, collection_name, point_id):
        objects = self.qdrant_client.retrieve(
            collection_name=collection_name, ids=[point_id]
        )
        if objects:
            return True
        return False

    def search_group_by(self, collection_name, query_vector, top_k, group_property):
        # Qdrant не поддерживает группировку результатов напрямую,
        # поэтому эта функция не может быть реализована без дополнительной обработки на стороне клиента.
        raise NotImplementedError("search_group_by is not supported in Qdrant")
