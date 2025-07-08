from typing import Dict, Type
from .models.base_embedder import BaseEmbedder
from .models.text_embedder import TextEmbedder
from .models.image_embedder import ImageEmbedder

REGISTERED_MODELS: Dict[str, Type[BaseEmbedder]] = {
    "all-MiniLM-L6-v2": TextEmbedder,
    "google/vit-base-patch16-224": ImageEmbedder,
    # "ViT-L/14": AnotherImageEmbedder, # Пример
}


LOADED_MODELS: Dict[str, BaseEmbedder] = {}


def get_embedder_instance(model_name: str) -> BaseEmbedder:
    """
    Возвращает инстанс эмбеддера по его имени.
    Осуществляет ленивую загрузку модели, если она еще не была загружена.
    """
    if model_name not in REGISTERED_MODELS:
        raise ValueError(f"Model '{model_name}' is not registered.")

    print(
        f"[DEBUG app/__init__] Attempting to create instance for model_name: '{model_name}' using class: {REGISTERED_MODELS.get(model_name)}"
    )
    if model_name == "google/vit-base-patch16-224":
        print(
            f"[DEBUG app/__init__] Matched image model: '{model_name}'. Will use ImageEmbedder."
        )

    if model_name not in LOADED_MODELS:
        print(f"Initializing model '{model_name}' for the first time...")
        EmbedderClass = REGISTERED_MODELS[model_name]
        LOADED_MODELS[model_name] = EmbedderClass(model_name=model_name)
        print(f"Model '{model_name}' initialized.")
    return LOADED_MODELS[model_name]


def get_default_text_model_name() -> str:
    for name, klass in REGISTERED_MODELS.items():
        if klass.mro()[1] == TextEmbedder or (
            hasattr(klass, "model_type") and klass.model_type == "text"
        ):
            if hasattr(klass("", model_type="text"), "model_type") and klass("", model_type="text").model_type == "text":  # type: ignore
                return name
    return "all-MiniLM-L6-v2"


def get_default_image_model_name() -> str:
    for name, klass in REGISTERED_MODELS.items():
        if klass.mro()[1] == ImageEmbedder or (
            hasattr(klass, "model_type") and klass.model_type == "image"
        ):
            if hasattr(klass("", model_type="image"), "model_type") and klass("", model_type="image").model_type == "image":  # type: ignore
                return name  
    return "google/vit-base-patch16-224"


def get_available_models_info():
    infos = []
    for name, klass in REGISTERED_MODELS.items():
        try:
            instance = get_embedder_instance(name)
            infos.append(instance.get_model_info())
        except Exception as e:
            print(f"Could not get info for model {name} due to init error: {e}")
            model_type_from_class = "unknown"
            if hasattr(
                klass, "model_type"
            ):
                model_type_from_class = klass.model_type
            elif "TextEmbedder" in str(klass):
                model_type_from_class = "text"
            elif "ImageEmbedder" in str(klass):
                model_type_from_class = "image"

            infos.append(
                {
                    "model_name": name,
                    "model_type": model_type_from_class,
                    "description": f"Error loading/initializing: {str(e)}",
                }
            )

    return infos


def preload_models():
    print("Preloading models...")
    for model_name in REGISTERED_MODELS.keys():
        try:
            get_embedder_instance(model_name)
        except Exception as e:
            print(f"Failed to preload model {model_name}: {e}")
    print("Model preloading complete.")
