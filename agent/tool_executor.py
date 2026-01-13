from typing import Dict, Callable, List

class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Callable] = dict()
        self.tool_schemas: List = []
    @property
    def available_tools(self):
        return self.tool_schemas
    def register_tool(self, tool_name: str, tool: Callable, tool_schema: Dict):
        self.tools[tool_name] = tool
        self.tool_schemas.append(tool_schema)
    def exec(self, tool_name: str, **args) -> str:
        return self.tools[tool_name](**args)