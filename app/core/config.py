from pathlib import Path
import os
import logging
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]

# Configure logger
logger = logging.getLogger("app.config")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# Load environment variables
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

logger.info(f"Loading environment from: {env_path}")

class Settings:
    def __init__(self):
        self.openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self.app_name: str = os.getenv("APP_NAME", "FastAPI LangChain OpenAI")
        self.debug: bool = os.getenv("DEBUG", "False").lower() == "true"

        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY is missing from .env")
        else:
            logger.info("OPENAI_API_KEY loaded successfully")

        logger.info(f"Using OpenAI model: {self.openai_model}")

settings = Settings()