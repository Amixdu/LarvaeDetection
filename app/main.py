from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(
    title="Larvae Detection API",
    description="Switchable Backend for Larvae Counting (CV vs DL)",
    version="0.1.0"
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    # Run with: python app/main.py
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)