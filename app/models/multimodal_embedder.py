import logging  # Import logging
from sentence_transformers import SentenceTransformer
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import requests
from typing import List, Union, Any
from app.models.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class MultimodalEmbedder(BaseEmbedder):
    """Класс для мультимодальных эмбеддингов с использованием SentenceTransformer (CLIP модели)."""

    description = (
        "Sentence Transformer CLIP model for multimodal (text/image) embeddings."
    )

    def __init__(
        self,
        model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        model_cache_dir: str = "./model_cache",
    ):
        super().__init__(
            model_name=model_name,
            model_type="multimodal",
            model_cache_dir=model_cache_dir,
        )
        self._dimension = 0

    def _load_model(self):
        logger.info(
            f"Loading multimodal model: {self.model_name} from cache: {self.model_cache_dir}..."
        )
        try:
            self.model = SentenceTransformer(
                self.model_name, cache_folder=self.model_cache_dir
            )
            self._dimension = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Multimodal model {self.model_name} loaded. Dimension: {self._dimension}"
            )
        except Exception as e:
            logger.error(
                f"Error loading SentenceTransformer model {self.model_name}: {e}",
                exc_info=True,
            )
            self.model = None
            raise

    def get_embedding(self, data: Union[str, Image.Image]) -> List[float]:
        if self.model is None:
            raise RuntimeError(
                f"Multimodal model {self.model_name} is not loaded properly."
            )

        try:
            if isinstance(data, str):
                embeddings = self.model.encode([data])
                embedding = embeddings[0]
            elif isinstance(data, Image.Image):
                embeddings = self.model.encode([data])
                embedding = embeddings[0]
            else:
                raise TypeError("Input data must be a string or a PIL Image.")

            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error during model encoding: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get embedding: {e}")

    @property
    def dimension(self) -> int:
        if self.model is None:
            raise RuntimeError(
                f"Multimodal model {self.model_name} is not loaded, dimension unknown."
            )
        if self._dimension == 0 and self.model is not None:
            try:
                self._dimension = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.error(f"Could not get embedding dimension: {e}", exc_info=True)
                raise RuntimeError(f"Could not determine model dimension: {e}")
        return self._dimension

    def _load_image_from_source(self, image_source: Union[bytes, str]) -> Image.Image:
        image_bytes: bytes
        if isinstance(image_source, str):  # Если URL
            try:
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image_bytes = response.content
            except requests.RequestException as e:
                logger.error(
                    f"Could not download image from URL: {image_source}. Error: {e}",
                    exc_info=True,
                )  # Use logger
                raise ValueError(
                    f"Could not download image from URL: {image_source}. Error: {e}"
                )
        elif isinstance(image_source, bytes):
            image_bytes = image_source
        else:
            raise TypeError("image_source must be bytes (file content) or str (URL).")

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            return image
        except UnidentifiedImageError as e:
            logger.error(
                f"Invalid image file content or unsupported format. Error: {e}",
                exc_info=True,
            )
            raise ValueError(
                f"Invalid image file content or unsupported format. Error: {e}"
            )
        except Exception as e:
            logger.error(f"Could not open image. Error: {e}", exc_info=True)
            raise ValueError(f"Could not open image. Error: {e}")
