from pymongo import MongoClient, errors


class MongoDBClient:
    def __init__(self, uri: str, db_name: str):
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            print("Connected to MongoDB")
        except errors.ConnectionError as e:
            print(f"Error connecting to MongoDB: {e}")

    def insert_one(self, collection_name: str, document: dict):
        try:
            collection = self.db[collection_name]
            result = collection.insert_one(document)
            print(f"Inserted document with _id: {result.inserted_id}")
        except Exception as e:
            print(f"Error inserting document: {e}")

    def find_one(self, collection_name: str, query: dict):
        try:
            collection = self.db[collection_name]
            document = collection.find_one(query)
            return document
        except Exception as e:
            print(f"Error finding document: {e}")
            return None

    def update_one(self, collection_name: str, query: dict, update: dict):
        try:
            collection = self.db[collection_name]
            result = collection.update_one(query, {"$set": update})
            print(
                f"Matched {result.matched_count} document(s) and modified {result.modified_count} document(s)"
            )
        except Exception as e:
            print(f"Error updating document: {e}")

    def delete_one(self, collection_name: str, query: dict):
        try:
            collection = self.db[collection_name]
            result = collection.delete_one(query)
            print(f"Deleted {result.deleted_count} document(s)")
        except Exception as e:
            print(f"Error deleting document: {e}")

    def iterate_videos(self, collection_name: str, offset: int = 0, limit: int = 1):
        try:
            collection = self.db[collection_name]
            cursor = collection.find({}).skip(offset)
            documents = list(cursor.limit(limit))
            cursor.close()
            for video in documents:
                yield video

        except Exception as e:
            print(f"Error iterating videos: {e}")

    def collection_size(self, collection_name: str):
        try:
            collection = self.db[collection_name]
            return collection.estimated_document_count()
        except Exception as e:
            print(f"Error getting collection size: {e}")
            return None
