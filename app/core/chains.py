from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from app.core.rag_agent import db


SYSTEM_PROMPT = """
You are Ganesh Resume Assistant.
Use ONLY the following context to answer.
If the answer is not present in context, refer chat history.
If still not found, say exactly: Data not available.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""


class LangChainManager:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
            temperature=0.7,
            streaming=False
        )
        self.chains = {}
        self.memories = {}

    def create_chain(self, chain_id: str):

        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=SYSTEM_PROMPT
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=db.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=True
        )

        self.chains[chain_id] = chain
        self.memories[chain_id] = memory
        return chain

    def get_chain(self, chain_id: str):
        if chain_id not in self.chains:
            return self.create_chain(chain_id)
        return self.chains[chain_id]

    def chat(self, chain_id: str, message: str) -> str:
        chain = self.get_chain(chain_id)
        result = chain.invoke({"question": message})
        return result["answer"]

    def clear_memory(self, chain_id: str):
        if chain_id in self.memories:
            self.memories[chain_id].clear()
