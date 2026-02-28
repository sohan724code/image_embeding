from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import clip
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io

# -------------------- TEXT EMBEDDING --------------------
from sentence_transformers import SentenceTransformer

app = FastAPI(title="ChatPilot Visual Search API (High Accuracy)")

# -------------------- DEVICE --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- CLIP (BETTER MODEL) --------------------
clip_model, preprocess = clip.load("ViT-B/16", device=device)
clip_model.eval()

# -------------------- YOLO --------------------
yolo = YOLO("yolov8n-seg.pt")

# -------------------- TEXT MODEL --------------------
# Loads once on startup, stays in memory — fast for every request
text_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# -------------------- SCHEMA --------------------
class Base64Image(BaseModel):
    image_base64: str

class TextInput(BaseModel):
    text: str

# =========================================================
# 🔵 CORE EMBEDDING (STRICT & NORMALIZED)
# =========================================================
def clip_embed(pil_img: Image.Image):
    img = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = clip_model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

# =========================================================
# 🟢 IMAGE UTILS
# =========================================================
def decode_base64(data: str) -> np.ndarray:
    try:
        data = data.split(",")[-1]
        img_bytes = base64.b64decode(data)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError
        return img
    except Exception:
        raise ValueError("INVALID_IMAGE")

def resize_if_large(img, max_size=1024):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def safe_segment(img):
    try:
        result = yolo(img, conf=0.4, verbose=False)[0]
        if result.masks is None:
            return img

        masks = result.masks.data.cpu().numpy()
        areas = masks.sum(axis=(1, 2))
        mask = masks[np.argmax(areas)]

        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return img

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        h, w = img.shape[:2]

        if (x2 - x1) < w * 0.2 or (y2 - y1) < h * 0.2:
            return img

        return img[y1:y2, x1:x2]
    except Exception:
        return img

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# =========================================================
# 🔥 FINAL FUSION LOGIC
# =========================================================
def dual_embedding(img):
    pil_full = cv2_to_pil(img)
    full_emb = clip_embed(pil_full)

    seg_img = safe_segment(img)
    pil_seg = cv2_to_pil(seg_img)
    seg_emb = clip_embed(pil_seg)

    fused = (0.6 * full_emb + 0.4 * seg_emb)
    fused = fused / np.linalg.norm(fused)

    return fused.tolist()

# =========================================================
# 🔵 INDEX API (PRODUCT CATALOG)
# =========================================================
@app.post("/index-product-image-base64")
async def index_product(payload: Base64Image):
    try:
        img = decode_base64(payload.image_base64)
        img = resize_if_large(img)
        embedding = dual_embedding(img)
        return {
            "embedding": embedding,
            "model": "CLIP ViT-B/16",
            "strategy": "full+segment fusion"
        }
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid product image")

# =========================================================
# 🟢 SEARCH API (USER QUERY IMAGE)
# =========================================================
@app.post("/search-image-base64")
async def search_image(payload: Base64Image):
    try:
        img = decode_base64(payload.image_base64)
        img = resize_if_large(img)

        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        image_quality = "low" if blur < 60 else "good"

        embedding = dual_embedding(img)

        return {
            "embedding": embedding,
            "image_quality": image_quality,
            "recommended_threshold": 0.82 if image_quality == "low" else 0.78
        }
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid search image")

# =========================================================
# 🟣 TEXT EMBEDDING API  ← NEW ENDPOINT
# =========================================================
@app.post("/text-embedding")
async def text_embedding(payload: TextInput):
    """
    Generates a 384-dim text embedding using all-MiniLM-L6-v2.
    Used for hybrid search alongside CLIP image embeddings.

    Input:  { "text": "Nike football boots red size 10" }
    Output: { "embedding": [0.123, -0.045, ...], "dimensions": 384, "model": "all-MiniLM-L6-v2" }
    """
    try:
        if not payload.text or len(payload.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="text field cannot be empty")

        # Truncate to 512 chars to stay within model token limits
        text = payload.text.strip()[:512]

        embedding = text_model.encode(
            text,
            normalize_embeddings=True,   # L2-normalised, ready for cosine similarity
            show_progress_bar=False
        ).tolist()

        return {
            "embedding":  embedding,
            "dimensions": len(embedding),
            "model":      "all-MiniLM-L6-v2"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text embedding failed: {str(e)}")

# =========================================================
# 🟣 SEARCH TEXT API  ← NEW ENDPOINT (for search queries)
# =========================================================
@app.post("/search-text")
async def search_text(payload: TextInput):
    """
    Same as /text-embedding but named for search context.
    Generates embedding for a user's text search query.

    Input:  { "text": "red running shoes" }
    Output: { "embedding": [...], "dimensions": 384 }
    """
    try:
        if not payload.text or len(payload.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="text field cannot be empty")

        text = payload.text.strip()[:512]

        embedding = text_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        return {
            "embedding":  embedding,
            "dimensions": len(embedding),
            "model":      "all-MiniLM-L6-v2"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search text embedding failed: {str(e)}")

# =========================================================
# 🟢 HEALTH
# =========================================================
@app.get("/")
def health():
    return {
        "status":            "ok",
        "clip_model":        "CLIP ViT-B/16",
        "text_model":        "all-MiniLM-L6-v2",
        "fusion":            "dual (full + segment)",
        "endpoints": {
            "index_image":   "POST /index-product-image-base64",
            "search_image":  "POST /search-image-base64",
            "index_text":    "POST /text-embedding",
            "search_text":   "POST /search-text"
        }
    }