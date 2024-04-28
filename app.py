from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,PromptTemplate
from llama_index.core.embeddings import resolve_embed_model

from custom_llm import Cloudflare_WorkersAI_LLM as llm
from custom_llm2 import Cloudflare_WorkersAI_LLM as code_llm

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

from prompts import context, code_parser_template
from code_reader import code_reader

parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# embed_model = resolve_embed_model("local:BAAI/bge-m3")            # 2.3GB
embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")   # 133MB
vector_index = VectorStoreIndex(documents, embed_model=embed_model, show_progress=True)

query_engine = vector_index.as_query_engine(llm=llm())

# result = query_engine.query("What are some of the routes in the api?")
# print(result)

tools=[
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description=("This gives documentation about code for an API."
                         "Use this for reading docs for the API."
                         "Use a detailed plain text question as input to the tool."
                        ),
        ),
    ),
    # code_reader,
]

agent = ReActAgent.from_tools(tools, llm=code_llm(), verbose=True, context=context)

# while (prompt := input("Enter a prompt (q to quit):")) != "q":
#     result = agent.chat(prompt)
#     print(result)
# result = agent.query("What are some of the routes in the api?")
result = agent.chat("read the contents of the test.py and give me the exact code back")
print(result)