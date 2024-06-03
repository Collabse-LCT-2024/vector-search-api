import weaviate.classes as wvc
from weaviate.classes.query import Filter
from weaviate.classes.config import Property, DataType
import weaviate
from weaviate.classes.query import MetadataQuery, GroupBy


class DataBaseClient:
    def __init__(self, WEAVIATE_HOST: str, index_name: str):
        self.index_name = index_name
        self.weaviate_client = weaviate.connect_to_custom(
            http_host=WEAVIATE_HOST,
            http_port=8080,
            http_secure=False,
            grpc_host=WEAVIATE_HOST,
            grpc_port=50051,
            grpc_secure=False,
        )
        self.collection = self.weaviate_client.collections.get(self.index_name)

    def search(self, query_vector, top_k):
        group_by = GroupBy(
            prop="external_id",  # group by this property
            objects_per_group=1800,
            number_of_groups=top_k,  # maximum number of groups
        )
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            group_by=group_by,
            limit=1000,
            return_metadata=wvc.query.MetadataQuery(certainty=True),
        )
        return response

    def drop_collection(self, index_name):
        self.weaviate_client.collections.delete(index_name)

    def create_collection(self, index_name: str):
        self.weaviate_client.collections.create(
            index_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE
            ),
            properties=[
                Property(name="external_id", data_type=DataType.TEXT),
                Property(name="link", data_type=DataType.TEXT),
            ],
        )

    def fetch_all(self, include_vector: bool = False):
        for item in self.collection.iterator(include_vector=include_vector):
            print(item)

    def is_present_by_id(self, unified_id: str):
        response = self.collection.query.fetch_objects(
            filters=Filter.by_property("external_id").equal(unified_id), limit=1
        )
        return len(response.objects) > 0
