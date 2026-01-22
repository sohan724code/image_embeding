from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import clip
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io

app = FastAPI(title="E-commerce Image Embedding API")

# -------------------- MODELS --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

yolo = YOLO("yolov8n-seg.pt")

# -------------------- SCHEMAS --------------------
class Base64Image(BaseModel):
    image_base64: str

# -------------------- CORE HELPERS --------------------

def clip_embedding(pil_img: Image.Image):
    image_input = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(image_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().tolist()[0]


def resize_if_large(img: np.ndarray, max_size=1024):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def safe_segment(img: np.ndarray):
    try:
        results = yolo(img, conf=0.4)[0]
        if results.masks is None:
            return img

        masks = results.masks.data.cpu().numpy()
        if len(masks) == 0:
            return img

        areas = [(m.sum(), i) for i, m in enumerate(masks)]
        _, idx = max(areas, key=lambda x: x[0])

        mask = (masks[idx] * 255).astype("uint8")
        product = cv2.bitwise_and(img, img, mask=mask)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return img

        return product[ys.min():ys.max(), xs.min():xs.max()]
    except Exception:
        return img


def read_file_to_cv2(file_bytes: bytes):
    np_img = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)

# -------------------- PRODUCT INDEXING (NO YOLO) --------------------

@app.post("/embed-product")
async def embed_product(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="INVALID_IMAGE")

    embedding = clip_embedding(pil_img)
    return {"embedding": embedding}

# -------------------- USER IMAGE SEARCH (YOLO + CLIP) --------------------

@app.post("/search-image")
async def search_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = read_file_to_cv2(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="INVALID_IMAGE")

    if img is None:
        raise HTTPException(status_code=400, detail="INVALID_IMAGE")

    img = resize_if_large(img)
    img = safe_segment(img)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    embedding = clip_embedding(pil_img)

    return {"embedding": embedding}

@app.post("/search-image-base64")
async def search_image_base64(payload: Base64Image):
    try:
        image_bytes = base64.b64decode(payload.image_base64)
        img = read_file_to_cv2(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="INVALID_BASE64_IMAGE")

    if img is None:
        raise HTTPException(status_code=400, detail="INVALID_BASE64_IMAGE")

    img = resize_if_large(img)
    img = safe_segment(img)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    embedding = clip_embedding(pil_img)

    return {"embedding": embedding}

# -------------------- HEALTH --------------------

@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "CLIP ViT-B/32 + YOLOv8-Seg",
        "device": device
    }
