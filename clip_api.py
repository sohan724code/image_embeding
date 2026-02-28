from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import clip
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from sentence_transformers import SentenceTransformer

app = FastAPI(title="ChatPilot Visual Search API")

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# MODELS  (loaded once at startup)
# ─────────────────────────────────────────────
clip_model, preprocess = clip.load("ViT-B/16", device=device)
clip_model.eval()

yolo = YOLO("yolov8n-seg.pt")

text_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ─────────────────────────────────────────────
# UNIVERSAL CANDIDATE LABELS
# Works for fashion, electronics, furniture,
# food, sports, beauty, toys — any product.
# CLIP scores the image against all of these
# to auto-generate a text description.
# ─────────────────────────────────────────────
CANDIDATE_LABELS = [

    # ── Product types ──────────────────────────
    "shirt", "t-shirt", "dress", "jacket", "coat", "hoodie", "sweater",
    "trousers", "jeans", "shorts", "skirt", "blouse", "suit", "uniform",
    "shoes", "boots", "sneakers", "sandals", "heels", "slippers", "loafers",
    "bag", "handbag", "backpack", "wallet", "purse", "luggage",
    "watch", "sunglasses", "belt", "hat", "cap", "scarf", "gloves",
    "jewelry", "necklace", "bracelet", "ring", "earrings",
    "phone", "laptop", "tablet", "headphones", "earbuds", "camera",
    "keyboard", "mouse", "monitor", "speaker", "charger", "cable",
    "sofa", "chair", "table", "desk", "bed", "shelf", "lamp",
    "kitchen appliance", "blender", "kettle", "coffee maker", "microwave",
    "toy", "game", "puzzle", "doll", "board game",
    "book", "notebook", "pen", "stationery",
    "cosmetics", "perfume", "skincare", "makeup", "shampoo",
    "food", "snack", "drink", "bottle", "packaging",
    "ball", "sports equipment", "fitness gear", "bicycle", "gym equipment",
    "tool", "hardware", "paint", "cleaning product",
    "plant", "pot", "garden item",
    "baby product", "baby clothes", "diaper",

    # ── Colors ─────────────────────────────────
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "grey", "brown", "beige", "gold", "silver",
    "multicolor", "transparent", "pastel",

    # ── Materials ──────────────────────────────
    "leather", "fabric", "cotton", "wool", "silk", "denim", "linen",
    "plastic", "metal", "wood", "glass", "rubber", "ceramic",

    # ── Style / Pattern ────────────────────────
    "striped", "checkered", "floral", "plain", "printed",
    "matte", "glossy", "textured",

    # ── Condition / State ──────────────────────
    "new", "packaged", "boxed", "unboxed",

    # ── Gender / Age ───────────────────────────
    "men", "women", "unisex", "kids", "baby",

    # ── Size signals ───────────────────────────
    "small", "large", "compact", "oversized",

    # ── Context ────────────────────────────────
    "indoor", "outdoor", "casual", "formal", "sport", "luxury", "budget",

    # ── Generic product signals ────────────────
    "product", "item", "accessory", "set", "bundle", "single item",
]


# ─────────────────────────────────────────────
# REQUEST SCHEMA
# ─────────────────────────────────────────────
class Base64Image(BaseModel):
    image_base64: str

class TextInput(BaseModel):
    text: str


# ─────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────
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

def resize_if_large(img, max_size: int = 1024):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def safe_segment(img):
    """YOLO segments the main object; falls back to full image if crop is too small."""
    try:
        result = yolo(img, conf=0.4, verbose=False)[0]
        if result.masks is None:
            return img
        masks = result.masks.data.cpu().numpy()
        mask  = masks[np.argmax(masks.sum(axis=(1, 2)))]
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

def clip_embed_image(pil_img: Image.Image) -> np.ndarray:
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = clip_model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def dual_image_embedding(img) -> list:
    """Fused embedding: 60% full image + 40% segmented object."""
    full_emb = clip_embed_image(cv2_to_pil(img))
    seg_emb  = clip_embed_image(cv2_to_pil(safe_segment(img)))
    fused    = 0.6 * full_emb + 0.4 * seg_emb
    fused    = fused / np.linalg.norm(fused)
    return fused.tolist()

def describe_image(img) -> str:
    """
    CLIP zero-shot classification:
    Scores all CANDIDATE_LABELS against the image,
    returns top matches as a comma-separated description.
    Works for any product category.
    """
    pil_img      = cv2_to_pil(img)
    img_tensor   = preprocess(pil_img).unsqueeze(0).to(device)
    text_tokens  = clip.tokenize(CANDIDATE_LABELS).to(device)

    with torch.inference_mode():
        img_feat  = clip_model.encode_image(img_tensor)
        txt_feat  = clip_model.encode_text(text_tokens)
        img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        scores    = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0].cpu().numpy()

    ranked = sorted(zip(CANDIDATE_LABELS, scores), key=lambda x: x[1], reverse=True)
    top    = [label for label, score in ranked if score > 0.012][:8]
    return ", ".join(top) if top else "product"

def embed_text(text: str) -> list:
    return text_model.encode(
        text.strip()[:512],
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

def process_image(img) -> dict:
    """
    Single unified function used by BOTH index and search endpoints.
    Returns image_embedding + text_embedding + image_description.
    """
    image_embedding   = dual_image_embedding(img)
    image_description = describe_image(img)
    text_embedding    = embed_text(image_description)

    return {
        "embedding":         image_embedding,    # 512-dim CLIP visual
        "text_embedding":    text_embedding,     # 384-dim from auto-description
        "image_description": image_description,  # human-readable
    }


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/index-product-image-base64")
async def index_product(payload: Base64Image):
    """
    Called by n8n product feed sync.
    Returns image_embedding + text_embedding for storing in DB.
    """
    try:
        img = decode_base64(payload.image_base64)
        img = resize_if_large(img)
        result = process_image(img)
        return {
            **result,
            "model":    "CLIP ViT-B/16",
            "strategy": "full+segment fusion"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed: {str(e)}")


@app.post("/search-image-base64")
async def search_image(payload: Base64Image):
    """
    Called when user sends a photo (e.g. Facebook chatbot).
    Returns the exact same embeddings as index endpoint
    so they can be compared directly against the DB.
    Also includes image quality + recommended threshold.
    """
    try:
        img = decode_base64(payload.image_base64)
        img = resize_if_large(img)

        # Image quality check
        blur          = cv2.Laplacian(img, cv2.CV_64F).var()
        image_quality = "low" if blur < 60 else "good"

        result = process_image(img)
        return {
            **result,
            "model":                 "CLIP ViT-B/16",
            "strategy":              "full+segment fusion",
            "image_quality":         image_quality,
            "recommended_threshold": 0.82 if image_quality == "low" else 0.78
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed: {str(e)}")


@app.post("/text-embedding")
async def text_embedding_endpoint(payload: TextInput):
    """For text-only or hybrid search queries."""
    try:
        if not payload.text or not payload.text.strip():
            raise HTTPException(status_code=400, detail="text cannot be empty")
        emb = embed_text(payload.text)
        return { "embedding": emb, "dimensions": len(emb), "model": "all-MiniLM-L6-v2" }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search-text")
async def search_text_endpoint(payload: TextInput):
    """Alias of /text-embedding for semantic clarity."""
    return await text_embedding_endpoint(payload)


@app.get("/")
def health():
    return {
        "status":     "ok",
        "clip_model": "CLIP ViT-B/16",
        "text_model": "all-MiniLM-L6-v2",
        "note":       "Universal — works for any product category",
        "endpoints": {
            "index":        "POST /index-product-image-base64",
            "search_image": "POST /search-image-base64",
            "search_text":  "POST /search-text",
            "text_embed":   "POST /text-embedding"
        }
    }