from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import clip
import io
import base64

app = FastAPI(title="CLIP Image Embedding API")

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ---------- Models ----------
class Base64Image(BaseModel):
    image_base64: str


# ---------- Helpers ----------
def image_to_embedding(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().tolist()[0]


# ---------- Endpoints ----------

# 1️⃣ Upload image as file (multipart/form-data)
@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    embedding = image_to_embedding(image)
    return {"embedding": embedding}


# 2️⃣ Send image as base64 (JSON)
@app.post("/embed-image-base64")
async def embed_image_base64(payload: Base64Image):
    try:
        image_bytes = base64.b64decode(payload.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    embedding = image_to_embedding(image)
    return {"embedding": embedding}


# ---------- Health Check ----------
@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "CLIP ViT-B/32",
        "device": device
    }
