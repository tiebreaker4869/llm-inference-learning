from llm import LLM
from search_tools import simple_websearch, simple_websearch_schema
from tool_executor import ToolExecutor
import json

class ReActAgent:
    def __init__(self, model: str = "gpt-5-mini-2025-08-07"):
        self.llm = LLM(model)
        self.instructions = "You are a helpful assistant."
        self.messages = [{"role": "developer", "content": self.instructions}]
        self.tool_executor = ToolExecutor()
        self.tool_executor.register_tool("simple_websearch", simple_websearch, simple_websearch_schema)
    def chat(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        final_response = None
        while True:
            response = self.llm.think(self.messages, tools=self.tool_executor.available_tools)
            final_response = response
            self.messages += response.output
            final_answer = True
            for item in response.output:
                if item.type == "function_call":
                    final_answer = False
                    args = json.loads(item.arguments)
                    print(f"[Tool Call] Calling {item.name} with args: {args}")
                    tool_output = self.tool_executor.exec(item.name, **args)
                    self.messages.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_output
                    })
            if final_answer:
                break
        return final_response.output_text

def main():
    agent = ReActAgent()
    while True:
        query = input("User: ")
        if query == "exit":
            break
        response = agent.chat(query)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()    