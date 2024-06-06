import os
from db import DataBaseClient
from fastapi import FastAPI
from dotenv import load_dotenv
from clip_client import Client
import translators as ts
from pymongo import MongoClient

load_dotenv(override=True)
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST")
CLIP_URL = os.environ.get("CLIP_URL")
MONOGO_URI = os.environ.get("MONGO_URI")

app = FastAPI(root_path="/api/v1")
print("PREACCELERATING TRANSLATION. MAY TAKE UP TO 5 MINUTES")
_ = ts.preaccelerate_and_speedtest()  # speed up translation.
client = DataBaseClient(WEAVIATE_HOST)
mongo = MongoClient(MONOGO_URI)
clip_client = Client(CLIP_URL)
db = mongo["storage"]
collection = db["videos"]
print("READY")


@app.get("/version")
def read_root():
    return {"version": "0.1"}


@app.post("/search/text")
def search_by_text(text: str, mode: int, top_k: int = 3):
    eng_text = ts.translate_text(
        query_text=text,
        translator="yandex",
        from_language="ru",
        to_language="en",
        if_use_preacceleration=True,
    )
    eng_text = text

    query_vector = clip_client.encode([eng_text])[0].tolist()
    if mode == 0:
        response = client.search_group_by(
            "FrameEmbeddings", query_vector, top_k, "external_id"
        )
    if mode == 1:
        response = client.search("MeanFrameEmbeddings", query_vector, top_k)
    if mode == 2:
        response = client.search("NormallyWeightedFrameEmbeddings", query_vector, top_k)

    result = []
    if mode == 0:
        for k, group in response.groups.items():
            print(group)
            print()
            external_id = group.objects[0].properties["external_id"]
            relevant_frames_count = group.number_of_objects
            link = collection.find_one({"uuid": external_id})["link"]
            result.append(
                {
                    "external_id": external_id,
                    "link": link,
                    "relevant_frames_count": relevant_frames_count,
                }
            )
    else:
        for obj in response.objects:
            external_id = obj.properties["external_id"]
            link = collection.find_one({"uuid": external_id})["link"]
            result.append({"external_id": external_id, "link": link})
    return result


@app.get("/encode")
def encode_text(text: str):
    eng_text = ts.translate_text(
        query_text=text,
        translator="yandex",
        from_language="ru",
        to_language="en",
        if_use_preacceleration=True,
    )
    text_embedding = clip_client.encode([eng_text])[0]
    return {"vector": text_embedding}
