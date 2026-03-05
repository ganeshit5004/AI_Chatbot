from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from app.core.config import settings
from pathlib import Path

os.environ["OPENAI_API_KEY"] = settings.openai_api_key  
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss_index"



db = FAISS.load_local(
    str(INDEX_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)


# def ask_ganesh(question: str):
#     return qa.run(question)
