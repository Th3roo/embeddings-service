from pydantic import BaseModel, Field
from typing import List, Optional, Union


class TextRequest(BaseModel):
    text: str = Field(..., example="Это пример текста для получения эмбеддинга.")
    model_name: Optional[str] = Field(
        None,
        example="all-MiniLM-L6-v2",
        description="Опционально: указать конкретную текстовую модель, если их несколько",
    )


class ImageUrlRequest(BaseModel):
    url: str = Field(..., example="https://example.com/image.jpg")
    model_name: Optional[str] = Field(
        None,
        example="ViT-B/32",
        description="Опционально: указать конкретную модель изображений, если их несколько",
    )



class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model_used: str
    dim: int


class ModelInfo(BaseModel):
    model_name: str
    model_type: str  # "text" или "image"
    description: Optional[str] = None


class AvailableModelsResponse(BaseModel):
    models: List[ModelInfo]
