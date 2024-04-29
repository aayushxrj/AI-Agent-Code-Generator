from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,PromptTemplate
from llama_index.core.embeddings import resolve_embed_model

from custom_llm import Cloudflare_WorkersAI_LLM as llm
from custom_llm2 import Cloudflare_WorkersAI_LLM as code_llm

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline

import ast
import os

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
    code_reader,  
]

agent = ReActAgent.from_tools(tools, llm=code_llm(), verbose=True, context=context)

class CodeOutput(BaseModel):
    code: str
    description: str
    filename:str

parser  = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm()])

while (prompt := input("Enter a prompt (q to quit):")) != "q":
    retries = 0

    while retries <3:
        try:
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:",""))
            break
        except Exception as e:
            print(f"Error occured, retry #{retries}: {e}")
            retries += 1
    if retries >=3:
        print("Unable to process request, try again...")
        continue
    
    print("Code generated: ", cleaned_json["code"])
    print("\n\nDescription: ", cleaned_json["description"])
    filename =  cleaned_json["filename"]
    print("\n\nFilename: ", filename)

    try:
        with open(os.path.join("output", filename), "w") as f:
            f.write(cleaned_json["code"])
            print(f"\n\nSaved file {filename} to ./output/ directory successfully.")
    except Exception as e:
        print(f"Error saving file...  {e}")


# result = agent.chat("What are some of the routes in the api?")
# result = agent.chat("read the contents of the test.py and give me the exact code back.")
# result = agent.chat("send a post request to make a new item using the api in python")
# result = agent.chat("read the contents of test.py and write a python script that calls the post endpoint to make a new item")
# print(result)