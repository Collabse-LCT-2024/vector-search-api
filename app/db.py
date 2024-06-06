from typing import List
import weaviate.classes as wvc
from weaviate.classes.query import Filter
from weaviate.classes.config import Property
import weaviate
from weaviate.classes.query import GroupBy


class DataBaseClient:
    def __init__(self, WEAVIATE_HOST):
        self.weaviate_client = weaviate.connect_to_custom(
            http_host=WEAVIATE_HOST,
            http_port=8080,
            http_secure=False,
            grpc_host=WEAVIATE_HOST,
            grpc_port=50051,
            grpc_secure=False,
        )

    def save_batch_embeddings(self, index_name, embeddings, properties: List[dict]):
        self.collection = self.weaviate_client.collections.get(index_name)
        question_objs = []
        for i, item in enumerate(embeddings):
            question_objs.append(
                wvc.data.DataObject(
                    properties=properties[i],
                    vector=item,
                )
            )
        self.collection.data.insert_many(question_objs)
        print("saved batch")

    def save_embedding(self, index_name, embedding, properties: dict):
        self.collection = self.weaviate_client.collections.get(index_name)
        self.collection.data.insert(properties=properties, vector=embedding)
        print("saved embedding")

    def search_group_by(self, index_name, query_vector, top_k, group_property):
        self.collection = self.weaviate_client.collections.get(index_name)
        group_by = GroupBy(
            prop=group_property,  # group by this property
            objects_per_group=1800,
            number_of_groups=top_k,  # maximum number of groups
        )
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            group_by=group_by,
            limit=1000,
            return_metadata=wvc.query.MetadataQuery(certainty=True),
        )
        print("SEARCHING")
        return response

    def search(self, index_name, query_vector, top_k):
        self.collection = self.weaviate_client.collections.get(index_name)
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(certainty=True),
        )
        return response

    def drop_collection(self, index_name):
        self.weaviate_client.collections.delete(index_name)

    def create_collection(self, name: str, properties: List[Property]):
        self.weaviate_client.collections.create(
            name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE
            ),
            properties=properties,
        )

    def fetch_all(self, index_name, include_vector: bool = False):
        self.collection = self.weaviate_client.collections.get(index_name)
        for item in self.collection.iterator(include_vector=include_vector):
            print(item)

    def is_present_by_id(self, index_name, equal_to, by_property: str):
        self.collection = self.weaviate_client.collections.get(index_name)
        response = self.collection.query.fetch_objects(
            filters=Filter.by_property(by_property).equal(equal_to), limit=1000
        )
        print(response.objects)
        return len(response.objects) > 0
