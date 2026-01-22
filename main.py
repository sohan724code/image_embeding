from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import base64
from openai import OpenAI

# -----------------------------------
# CONFIG
# -----------------------------------
OPENAI_API_KEY = ""
EMBEDDING_MODEL = "text-embedding-3-large"  # 1536 dims

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="OpenAI Image Embedding API (1536d)")


# ---------- Models ----------
class Base64Image(BaseModel):
    image_base64: str


# ---------- Helpers ----------
def image_to_embedding(image_bytes: bytes):
    """
    Convert image bytes → base64 → 1536-dim embedding
    """
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=image_base64
        )
        return response.data[0].embedding  # 1536 dims
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Endpoints ----------

# 1️⃣ Upload image as file (multipart/form-data)
@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        Image.open(io.BytesIO(image_bytes)).convert("RGB")  # validation only
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    embedding = image_to_embedding(image_bytes)
    return {
        "dimensions": len(embedding),
        "embedding": embedding
    }


# 2️⃣ Send image as base64 (JSON)
@app.post("/embed-image-base64")
async def embed_image_base64(payload: Base64Image):
    try:
        image_bytes = base64.b64decode(payload.image_base64)
        Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    embedding = image_to_embedding(image_bytes)
    return {
        "dimensions": len(embedding),
        "embedding": embedding
    }


# ---------- Health Check ----------
@app.get("/")
def health():
    return {
        "status": "ok",
        "model": EMBEDDING_MODEL,
        "dimensions": 1536
    }
