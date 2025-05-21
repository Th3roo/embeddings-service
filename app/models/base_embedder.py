from abc import ABC, abstractmethod
from typing import Any, List

class BaseEmbedder(ABC):
    """Абстрактный базовый класс для всех моделей эмбеддингов."""

    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type # "text" или "image"
        self.model = None
        self._load_model() # Загрузка модели при инициализации

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
            "description": getattr(self, 'description', "N/A")
        }

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Возвращает размерность выходного вектора эмбеддинга."""
        pass