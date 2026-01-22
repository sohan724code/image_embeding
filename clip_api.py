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

app = FastAPI(title="Product Image Analyzer API")

# -------------------- MODELS --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# YOLO SEGMENTATION
yolo = YOLO("yolov8n-seg.pt")

# -------------------- SCHEMAS --------------------

class Base64Image(BaseModel):
    image_base64: str  # base64 string WITHOUT data:image/... prefix


# -------------------- HELPERS --------------------

def image_to_embedding(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().tolist()[0]


def segment_product(img: np.ndarray):
    """
    Returns product-only image using segmentation.
    Falls back to original image if segmentation fails.
    """
    results = yolo(img, conf=0.4)[0]

    if results.masks is None:
        return img

    masks = results.masks.data.cpu().numpy()

    # pick largest mask
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


def process_image(img: np.ndarray):
    clean_img = segment_product(img)
    pil_img = Image.fromarray(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
    return image_to_embedding(pil_img)


# -------------------- API --------------------

# ✅ FILE UPLOAD (UNCHANGED)
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


# ✅ BASE64 INPUT (NEW)
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
        "model": "YOLOv8-Seg + CLIP ViT-B/32",
        "device": device
    }
