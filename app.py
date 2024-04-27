from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,PromptTemplate
from llama_index.core.embeddings import resolve_embed_model

from custom_llm import Cloudflare_WorkersAI_LLM as llm



parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# embed_model = resolve_embed_model("local:BAAI/bge-m3")            # 2.3GB
embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")   # 133MB
vector_index = VectorStoreIndex(documents, embed_model=embed_model, show_progress=True)

query_engine = vector_index.as_query_engine(llm=llm())

result = query_engine.query("What are some of the routes in the api?")
print(result)