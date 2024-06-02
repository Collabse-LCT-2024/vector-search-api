from pydantic import BaseModel


class Vector(BaseModel):
    external_id: str
    link: str
    vector: list[float] = None


class TextEmbedding(BaseModel):
    vector: list[float]
