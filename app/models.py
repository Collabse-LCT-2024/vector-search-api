from pydantic import BaseModel


class Vector(BaseModel):
    external_id: str
    link: str
    similar_frame_count: int
    max_distance: float


class TextEmbedding(BaseModel):
    vector: list[float]
