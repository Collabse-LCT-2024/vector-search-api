import os
from embeddings import embed_labse
from db import QdrantDBClient
from fastapi import FastAPI
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(override=True)
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST")
MONOGO_URI = os.environ.get("MONGO_URI")

app = FastAPI(root_path="/api/v1")
client = QdrantDBClient(WEAVIATE_HOST)
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
    group_db3: str = "",
):

    if not group_by:
        eng_text = text
        query_vector = embed_labse(eng_text).tolist()
        return client.search(collection, query_vector, top_k)
    else:
        eng_text = text
        query_vector = embed_labse(eng_text).tolist()
        response = client.experimental_groupby(
            query_vector, top_k, groub_db1, groub_db2, group_db3
        )
        return response
