from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # D:/Gen_ai

class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"

    app_name: str = "FastAPI LangChain OpenAI"
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

settings = Settings()
