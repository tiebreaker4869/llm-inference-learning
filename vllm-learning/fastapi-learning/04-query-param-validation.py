from fastapi import FastAPI, Query
from typing import Annotated

app = FastAPI()

@app.get("/items")
async def get_item(q: Annotated[str, Query(alias = "query", min_length=3, max_length=50)]):
    return {
        "query": q
    }