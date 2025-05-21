from typing import Dict, Type
from .base_embedder import BaseEmbedder
from .text_embedder import TextEmbedder #, AnotherTextEmbedder
from .image_embedder import ImageEmbedder #, AnotherImageEmbedder

# Реестр доступных моделей. Ключ - имя, которое будет использоваться в API.
# Значение - класс эмбеддера.
# Это упростит добавление новых моделей в будущем.
# Имена ключей должны быть уникальными!
REGISTERED_MODELS: Dict[str, Type[BaseEmbedder]] = {
    "all-MiniLM-L6-v2": TextEmbedder,
    # "paraphrase-multilingual-MiniLM-L12-v2": AnotherTextEmbedder, # Пример
    "ViT-B/32": ImageEmbedder,
    # "ViT-L/14": AnotherImageEmbedder, # Пример
}

# Словарь для хранения инстанцированных моделей (ленивая загрузка при первом обращении)
# Это гарантирует, что каждая модель загружается только один раз.
LOADED_MODELS: Dict[str, BaseEmbedder] = {}

def get_embedder_instance(model_name: str) -> BaseEmbedder:
    """
    Возвращает инстанс эмбеддера по его имени.
    Осуществляет ленивую загрузку модели, если она еще не была загружена.
    """
    if model_name not in REGISTERED_MODELS:
        raise ValueError(f"Model '{model_name}' is not registered.")

    if model_name not in LOADED_MODELS:
        print(f"Initializing model '{model_name}' for the first time...")
        EmbedderClass = REGISTERED_MODELS[model_name]
        # Передаем model_name в конструктор, т.к. классы могут быть инициализированы с разными model_name
        # например, TextEmbedder(model_name="custom-model")
        # Однако, здесь мы используем model_name как ключ, поэтому он должен совпадать с тем,
        # что ожидает конструктор класса по умолчанию.
        # Если конструктор TextEmbedder(model_name) всегда ожидает конкретное имя, то это нормально.
        # Если же мы хотим, чтобы ключ "my-custom-text-model" указывал на TextEmbedder("all-MiniLM-L6-v2"),
        # то нужно будет передавать `EmbedderClass(model_name_for_class_constructor)`
        LOADED_MODELS[model_name] = EmbedderClass(model_name=model_name)
        print(f"Model '{model_name}' initialized.")
    return LOADED_MODELS[model_name]

def get_default_text_model_name() -> str:
    # Возвращает имя первой зарегистрированной текстовой модели как модели по умолчанию
    for name, klass in REGISTERED_MODELS.items():
        # Проверяем, является ли класс наследником TextEmbedder или его тип "text"
        # Это немного грязно, лучше бы классы сами объявляли свой тип "text" или "image"
        # что мы и сделали в BaseEmbedder
        # Поэтому создадим временный инстанс, чтобы проверить тип (не загружая модель)
        # Или лучше, чтобы TextEmbedder и ImageEmbedder имели статическое поле model_type
        if klass.mro()[1] == TextEmbedder or (hasattr(klass, 'model_type') and klass.model_type == "text"): # Проверка базового класса
             # Для TextEmbedder(model_name="...") model_type будет "text"
            if hasattr(klass('', model_type='text'), 'model_type') and klass('', model_type='text').model_type == "text": # type: ignore
                return name # type: ignore
    return "all-MiniLM-L6-v2" # fallback

def get_default_image_model_name() -> str:
    # Аналогично для изображений
    for name, klass in REGISTERED_MODELS.items():
        if klass.mro()[1] == ImageEmbedder or (hasattr(klass, 'model_type') and klass.model_type == "image"):
             if hasattr(klass('', model_type='image'), 'model_type') and klass('', model_type='image').model_type == "image": # type: ignore
                return name # type: ignore
    return "ViT-B/32" # fallback

def get_available_models_info():
    infos = []
    for name, klass in REGISTERED_MODELS.items():
        # Создаем временный инстанс для получения информации, не загружая тяжелую модель
        # Предполагается, что model_name и model_type доступны без полной загрузки
        # Это немного неэффективно, если конструктор делает что-то тяжелое до _load_model
        # Для этого BaseEmbedder устанавливает model_name и model_type в __init__ до _load_model
        # Таким образом, get_model_info будет работать корректно.
        # Но полная загрузка произойдет, если модель еще не в LOADED_MODELS.
        # Чтобы избежать этого, можно сделать model_name и model_type атрибутами класса.

        # Лучше так:
        # model_type = "unknown"
        # if issubclass(klass, TextEmbedder): model_type = "text"
        # elif issubclass(klass, ImageEmbedder): model_type = "image"
        # description = getattr(klass, 'description', "N/A")
        # infos.append({"model_name": name, "model_type": model_type, "description": description})

        # С ленивой загрузкой:
        try:
            # Попытка получить инстанс (загрузит, если еще не загружен)
            # Это может быть медленно при первом запросе /models, но гарантирует актуальность
            instance = get_embedder_instance(name)
            infos.append(instance.get_model_info())
        except Exception as e:
            # Если модель не может быть загружена, отмечаем это
            print(f"Could not get info for model {name} due to init error: {e}")
            # Попробуем получить информацию из класса, если возможно
            model_type_from_class = "unknown"
            if hasattr(klass, 'model_type'): # Предполагаем, что это статический атрибут или свойство
                model_type_from_class = klass.model_type
            elif "TextEmbedder" in str(klass): model_type_from_class="text"
            elif "ImageEmbedder" in str(klass): model_type_from_class="image"

            infos.append({
                "model_name": name,
                "model_type": model_type_from_class,
                "description": f"Error loading/initializing: {str(e)}"
            })


    return infos

# Предзагрузка моделей при старте приложения (опционально, но рекомендуется для производительности)
# Чтобы избежать долгой первой загрузки при первом запросе.
def preload_models():
    print("Preloading models...")
    for model_name in REGISTERED_MODELS.keys():
        try:
            get_embedder_instance(model_name)
        except Exception as e:
            print(f"Failed to preload model {model_name}: {e}")
    print("Model preloading complete.")