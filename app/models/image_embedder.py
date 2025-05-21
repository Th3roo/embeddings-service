import torch
import clip
from PIL import Image
from io import BytesIO
import requests
from typing import List, Union
from .base_embedder import BaseEmbedder

class ImageEmbedder(BaseEmbedder):
    """Класс для эмбеддингов изображений с использованием CLIP."""
    description = "CLIP model for image embeddings."

    def __init__(self, model_name: str = "ViT-B/32"): # Один из стандартных CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_name=model_name, model_type="image")
        self._dimension = 0 # Будет установлено после загрузки модели


    def _load_model(self):
        print(f"Loading image model: {self.model_name} on device: {self.device}...")
        try:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            # Получаем размерность после загрузки
            # Для CLIP стандартный способ узнать размерность эмбеддинга изображения:
            # model.visual.output_dim или через тестовый прогон
            if hasattr(self.model.visual, 'output_dim'):
                 self._dimension = self.model.visual.output_dim
            else: # Fallback if output_dim is not directly available
                dummy_image = Image.new('RGB', (224, 224)) # CLIP typically uses 224x224
                image_input = self.preprocess(dummy_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    dummy_embedding = self.model.encode_image(image_input)
                self._dimension = dummy_embedding.shape[1]

            print(f"Image model {self.model_name} loaded. Dimension: {self._dimension}")
        except Exception as e:
            print(f"Error loading CLIP model {self.model_name}: {e}")
            self.model = None
            self.preprocess = None
            raise

    def get_embedding(self, image_source: Union[bytes, str]) -> List[float]:
        if self.model is None or self.preprocess is None:
            raise RuntimeError(f"Image model {self.model_name} is not loaded properly.")

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
        except Exception as e:
            raise ValueError(f"Could not open image. Error: {e}")

        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_input)

        return embedding.cpu().numpy().squeeze().tolist()

    @property
    def dimension(self) -> int:
        if self.model is None:
            raise RuntimeError(f"Image model {self.model_name} is not loaded, dimension unknown.")
        return self._dimension

# Пример добавления другой модели CLIP (если понадобится)
# class AnotherImageEmbedder(ImageEmbedder):
#     def __init__(self, model_name: str = "ViT-L/14"):
#         super().__init__(model_name=model_name)