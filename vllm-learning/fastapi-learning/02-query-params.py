from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def index():
    return {"message": "index page"}

# those not in path variables but in function parameters are interpreted as query parameters
@app.get("/get_entry")
async def get_paged_entry(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}