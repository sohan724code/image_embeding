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

app = FastAPI(title="ChatPilot Visual Search API")

# -------------------- DEVICE --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- CLIP --------------------
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# -------------------- YOLO (SEGMENTATION) --------------------
yolo = YOLO("yolov8n-seg.pt")

# -------------------- SCHEMAS --------------------
class Base64Image(BaseModel):
    image_base64: str

# =========================================================
# 🔵 CORE EMBEDDING (FINAL, STABLE)
# =========================================================

def image_to_embedding(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = clip_model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().tolist()[0]

# =========================================================
# 🟢 IMAGE HELPERS
# =========================================================

def resize_if_large(img: np.ndarray, max_size: int = 1024):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def safe_segment(img: np.ndarray):
    try:
        h, w = img.shape[:2]
        result = yolo(img, conf=0.45, verbose=False)[0]

        if result.masks is None:
            return img

        masks = result.masks.data.cpu().numpy()
        if len(masks) == 0:
            return img

        # pick largest object (main product)
        areas = masks.sum(axis=(1, 2))
        mask = masks[np.argmax(areas)]

        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0 or len(ys) == 0:
            return img

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # avoid tiny crops
        if (x2 - x1) < w * 0.35 or (y2 - y1) < h * 0.35:
            return img

        return img[y1:y2, x1:x2]

    except Exception:
        return img

def read_file_to_cv2(file_bytes: bytes):
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("INVALID_IMAGE")
    return img

def decode_base64(data: str) -> bytes:
    try:
        data = data.split(",")[-1]
        return base64.b64decode(data)
    except Exception:
        raise ValueError("INVALID_BASE64")

# =========================================================
# 🔵 INDEXING API (FOR PRODUCT UPLOAD)
# =========================================================
# 🔥 USE THIS FOR PRODUCT IMAGES ONLY

@app.post("/index-product-image-base64")
async def index_product_image(payload: Base64Image):
    try:
        image_bytes = decode_base64(payload.image_base64)
        img = read_file_to_cv2(image_bytes)

        # Preprocess
        img = resize_if_large(img)
        img = safe_segment(img)

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        embedding = image_to_embedding(pil_img)

        return {
            "embedding": embedding
        }

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid product image")

# =========================================================
# 🟢 SEARCH API (FOR USER IMAGES — YOUR USP)
# =========================================================
# 🔥 THIS IS YOUR MAIN COMMERCIAL ENDPOINT

@app.post("/search-image-base64")
async def search_image_base64(payload: Base64Image):
    try:
        image_bytes = decode_base64(payload.image_base64)
        img = read_file_to_cv2(image_bytes)

        # ---------- IMAGE TYPE ANALYSIS ----------
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean()

        image_type = "normal"
        if blur < 60:
            image_type = "video_frame"
        elif edge_density > 0.12:
            image_type = "screenshot"

        # ---------- PREPROCESS ----------
        img = resize_if_large(img)
        img = safe_segment(img)

        # Extra trim for screenshots
        if image_type == "screenshot":
            h, w = img.shape[:2]
            img = img[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        embedding = image_to_embedding(pil_img)

        return {
            "embedding": embedding,
            "image_type": image_type
        }

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid search image")

# =========================================================
# 🟢 HEALTH
# =========================================================

@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "CLIP ViT-B/32",
        "segmentation": "YOLOv8n-seg",
        "index_api": "/index-product-image-base64",
        "search_api": "/search-image-base64",
        "device": device
    }
