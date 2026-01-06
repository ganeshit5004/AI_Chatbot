from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from app.core.config import settings
from pathlib import Path

os.environ["OPENAI_API_KEY"] = settings.openai_api_key  
embeddings = OpenAIEmbeddings()
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss_index"


db = FAISS.load_local(
    str(INDEX_PATH),
    embeddings
)


# def ask_ganesh(question: str):
#     return qa.run(question)
