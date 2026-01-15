from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.core.config import settings

prompt = PromptTemplate(
    input_variables=["query"],
    template="""
Classify user query into one of:
- resume
- wiki
- chat

Return ONLY the label.

Query: {query}
Label:
"""
)

llm = ChatOpenAI(
    openai_api_key=settings.openai_api_key,
    model_name=settings.openai_model,
    temperature=0
)

def classify_intent(query: str) -> str:
    return llm.predict(prompt.format(query=query)).strip().lower()
