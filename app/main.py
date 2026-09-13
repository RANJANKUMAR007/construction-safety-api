from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(title="Construction Site Safety Detection & Reasoning API")

app.include_router(router)

@app.get("/")
def root():
    return {
        "status": "API is running",
        "available_endpoints": ["/detect", "/ask", "/docs"]
    }