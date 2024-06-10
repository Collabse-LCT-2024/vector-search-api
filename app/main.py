import os
from embeddings import embed_labse
from db import DataBaseClient
from fastapi import FastAPI
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(override=True)
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST")
CLIP_URL = os.environ.get("CLIP_URL")
MONOGO_URI = os.environ.get("MONGO_URI")

app = FastAPI(root_path="/api/v1")
client = DataBaseClient(WEAVIATE_HOST)
mongo = MongoClient(MONOGO_URI)
db = mongo["storage"]
collection = db["videos"]
print("READY")


@app.get("/version")
def read_root():
    return {"version": "0.1"}


@app.post("/search/text")
def search_by_text(
    text: str,
    collection: str,
    top_k: int = 3,
    group_by: bool = False,
    groub_db1: str = "",
    groub_db2: str = "",
):

    if not group_by:
        eng_text = text
        query_vector = embed_labse(eng_text).tolist()
        response = client.search(collection, query_vector, top_k)
        result = []
        for obj in response.objects:
            external_id = obj.properties["external_id"]
            link = collection.find_one({"uuid": external_id})["link"]
            vector = obj.vector
            result.append({"external_id": external_id, "link": link})
        return result
    else:
        eng_text = text
        query_vector = embed_labse(eng_text).tolist()
        response = client.experimental_groupby(
            query_vector, top_k, groub_db1, groub_db2
        )
        return response
