from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from app.core.rag_agent import db
from app.core.streaming import TokenStreamHandler

import asyncio


SYSTEM_PROMPT = """
You are Ganesh Resume Assistant.

Use ONLY the following context to answer the question.

Rules:
- If answer is present in context → answer clearly.
- If answer is not present in context → check chat history.
- If still not found → respond exactly with: Data not available.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""


class LangChainManager:

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.chains = {}
        self.memories = {}

    # Streaming LLM
    def _get_streaming_llm(self, handler):

        return ChatOpenAI(
            openai_api_key=self.api_key,
            model=self.model,
            temperature=0.7,
            streaming=True,
            callbacks=[handler],
        )

    # Create chain
    def create_chain(self, chain_id: str, llm=None):

        if llm is None:
            llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model=self.model,
                temperature=0.7
            )

        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=SYSTEM_PROMPT
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=True
        )

        self.chains[chain_id] = chain
        self.memories[chain_id] = memory

        return chain

    # Streaming chat
    async def chat_stream(self, chain_id: str, message: str):

        handler = TokenStreamHandler()
        llm = self._get_streaming_llm(handler)

        chain = self.create_chain(chain_id, llm)

        async def run_chain():
            await chain.ainvoke({"question": message})

        task = asyncio.create_task(run_chain())

        while True:

            token = await handler.queue.get()

            if token is None:
                break

            yield token

        await task

    # Get existing chain
    def get_chain(self, chain_id: str):

        if chain_id not in self.chains:
            return self.create_chain(chain_id)

        return self.chains[chain_id]

    # Normal chat
    def chat(self, chain_id: str, message: str) -> str:

        chain = self.get_chain(chain_id)

        result = chain.invoke({
            "question": message
        })

        return result.get("answer", "")

    # Clear conversation memory
    def clear_memory(self, chain_id: str):

        if chain_id in self.memories:
            self.memories[chain_id].clear()