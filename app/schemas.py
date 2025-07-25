from pydantic import BaseModel, Field
from typing import List, Optional, Union


class TextRequest(BaseModel):
    text: str = Field(..., example="This is an example text for getting an embedding.")
    model_name: Optional[str] = Field(
        None,
        example="all-MiniLM-L6-v2",
        description="Optional: specify a specific text model if there are several",
    )


class ImageUrlRequest(BaseModel):
    url: str = Field(..., example="https://example.com/image.jpg")
    model_name: Optional[str] = Field(
        None,
        example="ViT-B/32",
        description="Optional: specify a specific image model if there are several",
    )


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model_used: str
    dim: int


class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    description: Optional[str] = None


class AvailableModelsResponse(BaseModel):
    models: List[ModelInfo]
