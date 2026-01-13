from openai import OpenAI
from openai.types.responses import Response
from typing import Dict, List

class LLM:
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model
    def think(self, messages: List[Dict[str, str]], temperature: float = 1.0, tools = []) -> Response:
        try:
            response = self.client.responses.create(model=self.model, temperature=temperature, input=messages, tools=tools)
            return response
        except Exception as e:
            print(f"Error: fail to call LLM API. {e}")