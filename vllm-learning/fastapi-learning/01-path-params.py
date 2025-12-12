from fastapi import FastAPI

app = FastAPI()

@app.get("/user/{user_id}")
async def get_user_id(user_id: int):
    return {"user_id": user_id}