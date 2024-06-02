import os
from db import DataBaseClient
from fastapi import FastAPI
from dotenv import load_dotenv
from clip_client import Client
from models import TextEmbedding, Vector
import translators as ts

load_dotenv(override=True)
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST")
CLIP_URL = os.environ.get("CLIP_URL")

app = FastAPI()
_ = ts.preaccelerate_and_speedtest()  # speed up translation.
client = DataBaseClient(WEAVIATE_HOST, "EmbeddingsIndex")
clip_client = Client(CLIP_URL)


@app.get("/health")
def read_root():
    return "OK"


@app.post("/items/search_by_vector")
def search_by_vector(
    query_vector: list[float], top_k: int = 3, include_vector: bool = False
) -> list[Vector]:
    response = client.search(query_vector, top_k)
    result = []
    for o in response.objects:
        if not include_vector:
            result.append(
                Vector(
                    external_id=o.properties["external_id"],
                    link=o.properties["link"],
                )
            )
        else:
            result.append(
                Vector(
                    external_id=o.properties["external_id"],
                    link=o.properties["link"],
                    vector=o.vector,
                )
            )
    return result


@app.post("/items/search_by_text")
def search_by_text(
    text: str, top_k: int = 3, include_vector: bool = False
) -> list[Vector]:
    eng_text = ts.translate_text(
        query_text=text,
        translator="yandex",
        from_language="ru",
        to_language="en",
        if_use_preacceleration=True,
    )
    text_embedding = clip_client.encode([eng_text])[0]
    response = client.search(text_embedding.tolist(), top_k)
    result = []
    for o in response.objects:
        if not include_vector:
            result.append(
                Vector(
                    external_id=o.properties["external_id"],
                    link=o.properties["link"],
                )
            )
        else:
            result.append(
                Vector(
                    external_id=o.properties["external_id"],
                    link=o.properties["link"],
                    vector=o.vector,
                )
            )
    return result


@app.get("/encode_text")
def encode_text(text: str) -> TextEmbedding:
    text_embedding = clip_client.encode([text])[0]
    return TextEmbedding(vector=text_embedding)
