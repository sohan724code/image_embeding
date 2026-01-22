from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import clip
import io
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI(title="Product Image Analyzer API")

# ---------- Load Models ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# YOLO
yolo_model = YOLO("yolov8n.pt")

# ---------- Models ----------
class Base64Image(BaseModel):
    image_base64: str


# ---------- Helpers ----------

def image_to_embedding(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().tolist()[0]


def is_screenshot(img: np.ndarray) -> bool:
    """Simple screenshot / UI detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    edge_density = edges.mean()
    return edge_density > 30  # tune if needed


def crop_largest_product(img: np.ndarray):
    results = yolo_model(img)[0]
    h, w, _ = img.shape

    best_area = 0
    best_crop = None
    best_class = None

    for box in results.boxes:
        conf = float(box.conf)
        cls_id = int(box.cls)

        if conf < 0.7:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area > best_area:
            best_area = area
            best_crop = img[y1:y2, x1:x2]
            best_class = yolo_model.names[cls_id]

    # Reject small objects
    if best_crop is None or best_area < 0.25 * (h * w):
        return None, None

    return best_crop, best_class


def img_to_base64(img: np.ndarray):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


# ---------- API Endpoints ----------

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Screenshot / UI detection
    if is_screenshot(img):
        return {
            "status": "REJECTED",
            "reason": "SCREENSHOT_OR_UI"
        }

    # YOLO product crop
    cropped, product_type = crop_largest_product(img)

    if cropped is None:
        return {
            "status": "REJECTED",
            "reason": "NO_CLEAR_PRODUCT"
        }

    # CLIP embedding
    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    embedding = image_to_embedding(pil_img)

    return {
        "status": "OK",
        "confidence": 0.9,
        "product_type": product_type,
        "cropped_image_base64": img_to_base64(cropped),
        "clip_embedding": embedding
    }


# ---------- Optional: Base64 Input ----------

@app.post("/analyze-image-base64")
async def analyze_image_base64(payload: Base64Image):
    try:
        image_bytes = base64.b64decode(payload.image_base64)
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    if is_screenshot(img):
        return {
            "status": "REJECTED",
            "reason": "SCREENSHOT_OR_UI"
        }

    cropped, product_type = crop_largest_product(img)

    if cropped is None:
        return {
            "status": "REJECTED",
            "reason": "NO_CLEAR_PRODUCT"
        }

    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    embedding = image_to_embedding(pil_img)

    return {
        "status": "OK",
        "confidence": 0.9,
        "product_type": product_type,
        "cropped_image_base64": img_to_base64(cropped),
        "clip_embedding": embedding
    }


# ---------- Health ----------
@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "YOLOv8 + CLIP ViT-B/32",
        "device": device
    }
