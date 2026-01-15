from langchain.callbacks.base import BaseCallbackHandler
import asyncio

class TokenStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put_nowait(token)

    def on_llm_end(self, *args, **kwargs):
        self.queue.put_nowait(None)
