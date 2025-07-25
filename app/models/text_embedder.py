from sentence_transformers import SentenceTransformer
from typing import List
from .base_embedder import BaseEmbedder


class TextEmbedder(BaseEmbedder):
    """Class for text embeddings using SentenceTransformer."""

    description = "Sentence Transformer model for text embeddings."

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_cache_dir: str = "./model_cache",
    ):
        super().__init__(
            model_name=model_name, model_type="text", model_cache_dir=model_cache_dir
        )
        self._dimension = 0

    def _load_model(self):
        print(
            f"Loading text model: {self.model_name} from cache: {self.model_cache_dir}..."
        )
        try:
            self.model = SentenceTransformer(
                self.model_name, cache_folder=self.model_cache_dir
            )
            dummy_embedding = self.model.encode("test")
            self._dimension = dummy_embedding.shape[0]
            print(f"Text model {self.model_name} loaded. Dimension: {self._dimension}")
        except Exception as e:
            print(f"Error loading SentenceTransformer model {self.model_name}: {e}")
            self.model = None
            raise

    def get_embedding(self, text: str) -> List[float]:
        if self.model is None:
            raise RuntimeError(f"Text model {self.model_name} is not loaded properly.")
        embedding = self.model.encode(text)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        if self.model is None:

            raise RuntimeError(
                f"Text model {self.model_name} is not loaded, dimension unknown."
            )
        return self._dimension


