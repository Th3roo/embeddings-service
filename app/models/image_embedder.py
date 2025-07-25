import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from io import BytesIO
import requests
from typing import List, Union
from .base_embedder import BaseEmbedder

# TODO FIX MODEL(google/vit-base-patch16-224) LOADING or swap to another model
class ImageEmbedder(BaseEmbedder):
    """Класс для эмбеддингов изображений с использованием Hugging Face Transformers."""
    description = "Hugging Face ViT model for image embeddings."

    def __init__(self, model_name: str = "google/vit-base-patch16-224", model_cache_dir: str = "./model_cache"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_name=model_name, model_type="image", model_cache_dir=model_cache_dir)
        self._dimension = 0
        self.processor = None

    def _load_model(self):
        print(f"Loading image model: {self.model_name} on device: {self.device} from cache: {self.model_cache_dir}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
            
            if hasattr(self.model.config, 'hidden_size'):
                self._dimension = self.model.config.hidden_size
            else:
                dummy_image = Image.new('RGB', (self.processor.size['height'], self.processor.size['width']))
                inputs = self.processor(images=dummy_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    dummy_embedding = outputs.last_hidden_state
                self._dimension = dummy_embedding.shape[-1]

            print(f"Image model {self.model_name} loaded. Dimension: {self._dimension}")
        except Exception as e:
            print(f"Error loading Hugging Face model {self.model_name}: {e}")
            self.model = None
            self.processor = None
            raise

    def get_embedding(self, image_source: Union[bytes, str]) -> List[float]:
        if self.model is None or self.processor is None:
            raise RuntimeError(f"Image model {self.model_name} is not loaded properly.")

        image_bytes: bytes
        if isinstance(image_source, str):
            try:
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image_bytes = response.content
            except requests.RequestException as e:
                raise ValueError(f"Could not download image from URL: {image_source}. Error: {e}")
        elif isinstance(image_source, bytes):
            image_bytes = image_source
        else:
            raise TypeError("image_source must be bytes (file content) or str (URL).")

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not open image. Error: {e}")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.cpu().numpy().squeeze().tolist()

    @property
    def dimension(self) -> int:
        if self._dimension == 0 and self.model is not None:
            if hasattr(self.model.config, 'hidden_size'):
                self._dimension = self.model.config.hidden_size
            else:
                 raise RuntimeError(f"Image model {self.model_name} is loaded, but dimension could not be determined from config.")
        elif self.model is None and self._dimension == 0:
             raise RuntimeError(f"Image model {self.model_name} is not loaded, dimension unknown.")
        return self._dimension

# Пример НЕ УДАЛЯТЬ, может пригодиться для других моделей
# class AnotherImageEmbedder(ImageEmbedder):
#     def __init__(self, model_name: str = "facebook/deit-base-distilled-patch16-224", model_cache_dir: str = "./model_cache"): # Пример другой модели
#         super().__init__(model_name=model_name, model_cache_dir=model_cache_dir)