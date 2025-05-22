from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO
import requests
from typing import List, Union, Any
from app.models.base_embedder import BaseEmbedder # Assuming this is the correct path

class MultimodalEmbedder(BaseEmbedder):
    """Класс для мультимодальных эмбеддингов с использованием SentenceTransformer (CLIP модели)."""
    description = "Sentence Transformer CLIP model for multimodal (text/image) embeddings."

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1", model_cache_dir: str = "./model_cache"):
        super().__init__(model_name=model_name, model_type="multimodal", model_cache_dir=model_cache_dir)
        self._dimension = 0 # Будет установлено после загрузки модели

    def _load_model(self):
        print(f"Loading multimodal model: {self.model_name} from cache: {self.model_cache_dir}...")
        try:
            # У SentenceTransformer нет 'device' параметра при инициализации, он определяется автоматически
            # или можно переместить модель на устройство после загрузки: self.model.to(self.device)
            self.model = SentenceTransformer(self.model_name, cache_folder=self.model_cache_dir)
            # Получаем размерность после загрузки
            # Для CLIP моделей, размерность эмбеддинга фиксирована и известна.
            # Можно получить её через self.model.get_sentence_embedding_dimension()
            self._dimension = self.model.get_sentence_embedding_dimension()
            print(f"Multimodal model {self.model_name} loaded. Dimension: {self._dimension}")
        except Exception as e:
            print(f"Error loading SentenceTransformer model {self.model_name}: {e}")
            self.model = None # Убедимся, что модель не используется, если загрузка не удалась
            raise

    def get_embedding(self, data: Union[str, Image.Image]) -> List[float]:
        if self.model is None:
            raise RuntimeError(f"Multimodal model {self.model_name} is not loaded properly.")
        
        # Метод encode у SentenceTransformer для CLIP моделей может принимать строки или PIL Image
        embedding = self.model.encode(data)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        if self.model is None:
            raise RuntimeError(f"Multimodal model {self.model_name} is not loaded, dimension unknown.")
        if self._dimension == 0 and self.model is not None: # Если вдруг не установилось в _load_model
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension

    # Вспомогательный метод для загрузки изображения из URL или байтов, если нужно будет передавать не PIL.Image
    def _load_image_from_source(self, image_source: Union[bytes, str]) -> Image.Image:
        image_bytes: bytes
        if isinstance(image_source, str): # Если URL
            try:
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image_bytes = response.content
            except requests.RequestException as e:
                raise ValueError(f"Could not download image from URL: {image_source}. Error: {e}")
        elif isinstance(image_source, bytes): # Если байты файла
            image_bytes = image_source
        else:
            raise TypeError("image_source must be bytes (file content) or str (URL).")

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Could not open image. Error: {e}")
