from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings

import os, requests
from dotenv import load_dotenv
load_dotenv()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")

model = "@cf/meta/llama-2-7b-chat-int8"

class Cloudflare_WorkersAI_LLM(CustomLLM):
    context_window: int = 4096
    num_output: int = 256
    model_name: str = model

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{model}",
            headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
            json={"messages": [
                {"role": "system", "content": "You are a Helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]}
        )
        inference = response.json()
        res = inference["result"]["response"]
        return CompletionResponse(text=str(res))

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{model}",
            headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
            json={"messages": [
                {"role": "system", "content": "You are a Helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]}
        )
        inference = response.json()
        res = inference["result"]["response"]
        res = str(res)
        response = ""
        for token in res:
            response += token
            yield CompletionResponse(text=response, delta=token)