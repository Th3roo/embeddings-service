import os
from abc import ABC, abstractmethod
from typing import Any, List


class BaseEmbedder(ABC):
    """Абстрактный базовый класс для всех моделей эмбеддингов."""

    def __init__(
        self, model_name: str, model_type: str, model_cache_dir: str = "./model_cache"
    ):
        self.model_name = model_name
        self.model_type = model_type  # "text" или "image"
        self.model_cache_dir = model_cache_dir
        self.model = None

        if self.model_cache_dir:
            os.makedirs(self.model_cache_dir, exist_ok=True)
            print(f"Models will be cached in: {os.path.abspath(self.model_cache_dir)}")
        else:
            print(
                "Model cache directory not specified. Models will be downloaded to default Hugging Face cache."
            )

        self._load_model()  # Загрузка модели при инициализации

    @abstractmethod
    def _load_model(self):
        """Загружает конкретную модель. Должна быть реализована в подклассах."""
        pass

    @abstractmethod
    def get_embedding(self, data: Any) -> List[float]:
        """Генерирует эмбеддинг для входных данных. Должна быть реализована в подклассах."""
        pass

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "description": getattr(self, "description", "N/A"),
        }

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Возвращает размерность выходного вектора эмбеддинга."""
        pass
