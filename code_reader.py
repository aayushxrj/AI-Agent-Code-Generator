from llama_index.core.tools import FunctionTool
import os


# For better function defnition 
# from pydantic import Field
#def get_weather(
#    location: str = Field(
#        description="A city name and state, formatted like '<name>, <state>'"
#    ),
#) -> str:
#    """Usfeful for getting the weather for a given location."""
#    ...

#tool = FunctionTool.from_defaults(get_weather)


def code_reader_func(file_name):
    path = os.path.join("data", file_name)
    try:
        with open(path, "r") as f:
            content = f.read()
            return {"file_content": content}
    except Exception as e:
        return {"error": str(e)}


code_reader = FunctionTool.from_defaults(
    fn=code_reader_func,
    name="code_reader",
    description=("This tool can read the contents of code files and return their results."
    "Use this when you need to read the contents of a file."),
)