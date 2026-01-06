from pydantic import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"

    app_name: str = "FastAPI LangChain OpenAI"
    debug: bool = False

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

settings = Settings()
