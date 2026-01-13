from tavily import TavilyClient
import os
from pydantic import BaseModel, ConfigDict

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MAX_RESULTS = 5

class SimpleSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str

def simple_websearch(query: str):
    response = client.search(query)
    results = response["results"][:MAX_RESULTS]
    results = [f"title: {result['title']}\ncontent: {result['content']}\n" for result in results]
    return "\n".join(results)

simple_websearch_schema = {
    "type": "function",
    "name": "simple_websearch",
    "description": "search over web using tavily search api.",
    "parameters": SimpleSearchArgs.model_json_schema(),
    "strict": True,
}