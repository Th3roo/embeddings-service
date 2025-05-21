# app/core/config.py

APP_VERSION = "0.2.0" # Example setting, matches current app version in main.py
SERVICE_NAME = "Embeddings Service"
MODEL_CACHE_DIR = "./model_cache" # Documenting the default, as used in BaseEmbedder and docker-compose

# In the future, this could load settings from environment variables using Pydantic's BaseSettings
# from pydantic_settings import BaseSettings
#
# class Settings(BaseSettings):
# APP_VERSION: str = "0.2.0"
#     SERVICE_NAME: str = "Embeddings Service"
#     MODEL_CACHE_DIR: str = "./model_cache"
#     VALID_API_KEYS: List[str] = [] # Example: could load from .env
#
#     # other settings...
#
#     class Config:
#         env_file = ".env"
#         env_file_encoding = 'utf-8'
#
# settings = Settings()

# For now, we'll keep it simple. These values are mostly for documentation
# or could be imported by other modules if needed, but are not actively
# configuring the application behavior that isn't already hardcoded or
# set by default in other places.
# The actual API keys are loaded from .env in app/auth.py
# The model cache directory is passed as a default argument in BaseEmbedder.
