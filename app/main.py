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

app = FastAPI(root_path="/api/v1")
print("PREACCELERATING TRANSLATION. MAY TAKE UP TO 5 MINUTES")
_ = ts.preaccelerate_and_speedtest()  # speed up translation.
client = DataBaseClient(WEAVIATE_HOST, "EmbeddingsIndex")
clip_client = Client(CLIP_URL)


@app.get("/version")
def read_root():
    return {"version": "0.1"}


@app.post("/search/vector")
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
                )
            )
    return result


@app.post("/search/text")
def search_by_text(text: str, top_k: int = 3) -> list[Vector]:
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
    for k, v in response.groups.items():  # View by group
        result.append(
            Vector(
                external_id=v.objects[0].properties["external_id"],
                link=v.objects[0].properties["link"],
                similar_frame_count=v.number_of_objects,
                max_distance=v.max_distance,
            )
        )
    return result


@app.get("/encode")
def encode_text(text: str) -> TextEmbedding:
    eng_text = ts.translate_text(
        query_text=text,
        translator="yandex",
        from_language="ru",
        to_language="en",
        if_use_preacceleration=True,
    )
    text_embedding = clip_client.encode([eng_text])[0]
    return TextEmbedding(vector=text_embedding)
