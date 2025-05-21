from sentence_transformers import SentenceTransformer
from typing import List
from .base_embedder import BaseEmbedder

class TextEmbedder(BaseEmbedder):
    """Класс для эмбеддингов текста с использованием SentenceTransformer."""
    description = "Sentence Transformer model for text embeddings."

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name=model_name, model_type="text")
        self._dimension = 0 # Будет установлено после загрузки модели

    def _load_model(self):
        print(f"Loading text model: {self.model_name}...")
        try:
            self.model = SentenceTransformer(self.model_name)
            # Получаем размерность после загрузки
            dummy_embedding = self.model.encode("test")
            self._dimension = dummy_embedding.shape[0]
            print(f"Text model {self.model_name} loaded. Dimension: {self._dimension}")
        except Exception as e:
            print(f"Error loading SentenceTransformer model {self.model_name}: {e}")
            self.model = None # Убедимся, что модель не используется, если загрузка не удалась
            raise

    def get_embedding(self, text: str) -> List[float]:
        if self.model is None:
            raise RuntimeError(f"Text model {self.model_name} is not loaded properly.")
        embedding = self.model.encode(text)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        if self.model is None:
            # Можно либо вызывать ошибку, либо вернуть 0 или значение по умолчанию
            # Для согласованности с _load_model, где устанавливается _dimension
            raise RuntimeError(f"Text model {self.model_name} is not loaded, dimension unknown.")
        return self._dimension

# Пример добавления другой текстовой модели (если понадобится)
# class AnotherTextEmbedder(TextEmbedder):
#     def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
#         super().__init__(model_name=model_name)