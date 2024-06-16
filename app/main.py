import os
from embeddings import embed_labse
from db import QdrantDBClient
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv(override=True)
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST")

app = FastAPI()
client = QdrantDBClient(WEAVIATE_HOST)
qdrant_collection = "WeightedMeanFinal"


@app.get("/search")
def search_by_text(text: str):
    query_vector = embed_labse(text).tolist()
    response = client.search(qdrant_collection, query_vector, 10)
    res = []
    for obj in response:
        link = obj.payload["link"]
        des = obj.payload["description"]
        res.append({"link": link, "description": des})
    return res
