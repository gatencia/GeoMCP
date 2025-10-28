from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "Hello from GeoMCP!"}