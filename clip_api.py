from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import clip
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = FastAPI(title="Product Image Analyzer API")

# -------------------- MODELS --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# YOLO SEGMENTATION
yolo = YOLO("yolov8n-seg.pt")

# -------------------- SCHEMA --------------------
class Base64Image(BaseModel):
    image_base64: str  # raw base64 only

# -------------------- HELPERS --------------------

def image_to_embedding(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(image_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().tolist()[0]


def resize_if_large(img: np.ndarray, max_size: int = 1024):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def segment_product(img: np.ndarray):
    """
    Safe YOLO segmentation.
    NEVER crashes. Always returns an image.
    """
    try:
        results = yolo(img, conf=0.4)[0]

        if results.masks is None:
            return img

        masks = results.masks.data.cpu().numpy()
        if len(masks) == 0:
            return img

        # choose largest mask
        areas = [(mask.sum(), i) for i, mask in enumerate(masks)]
        _, idx = max(areas, key=lambda x: x[0])

        mask = (masks[idx] * 255).astype("uint8")
        product = cv2.bitwise_and(img, img, mask=mask)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return img

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        return product[y1:y2, x1:x2]

    except Exception as e:
        # absolute safety
        print("Segmentation error:", e)
        return img


def process_image(img: np.ndarray):
    img = resize_if_large(img)
    img = segment_product(img)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return image_to_embedding(pil_img)

# -------------------- API --------------------

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="INVALID_IMAGE")

    if img is None:
        raise HTTPException(status_code=400, detail="INVALID_IMAGE")

    embedding = process_image(img)

    return {
        "embedding": embedding
    }


@app.post("/analyze-image-base64")
async def analyze_image_base64(payload: Base64Image):
    try:
        image_bytes = base64.b64decode(payload.image_base64)
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="INVALID_BASE64_IMAGE")

    if img is None:
        raise HTTPException(status_code=400, detail="INVALID_BASE64_IMAGE")

    embedding = process_image(img)

    return {
        "embedding": embedding
    }

# -------------------- HEALTH --------------------

@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "YOLOv8-Seg + CLIP ViT-B/32 (STABLE)",
        "device": device
    }
