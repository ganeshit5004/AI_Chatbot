from pathlib import Path
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.core.config import settings

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ganesh_profile.txt"
INDEX_PATH = BASE_DIR / "faiss_index"

os.environ["OPENAI_API_KEY"] = settings.openai_api_key

print("Loading profile from:", DATA_PATH)

loader = TextLoader(str(DATA_PATH), encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print("Chunks created:", len(chunks))

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

db = FAISS.from_documents(chunks, embeddings)
db.save_local(str(INDEX_PATH))

print("✅ FAISS INDEX CREATED SUCCESSFULLY at", INDEX_PATH)