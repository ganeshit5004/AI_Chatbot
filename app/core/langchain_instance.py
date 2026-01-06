from app.core.config import settings
from app.core.chains import LangChainManager

langchain_manager = LangChainManager(
    api_key=settings.openai_api_key,
    model=settings.openai_model
)
