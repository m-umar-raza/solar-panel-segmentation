"""
FastAPI inference service for solar panel segmentation.

Accepts an aerial image, runs U-Net segmentation,
returns panel coverage stats and predicted mask.
"""

import base64
import io
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.models.unet import build_unet

# --- app setup ---
app = FastAPI(
    title="Solar Panel Segmentation API",
    description="Detect and segment solar panels in aerial imagery using U-Net.",
    version="1.0.0",
)

# allow cross-origin requests (needed for browser-based demos)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- model loading ---
# model is loaded once at startup, not on every request
_model = None
_device = None


def get_model():
    """Load model on first call, return cached model on subsequent calls."""
    global _model, _device

    if _model is not None:
        return _model, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # find model weights
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        # try relative to this file
        model_path = Path(__file__).parent.parent.parent / "models" / "best_model.pth"

    if not model_path.exists():
        raise RuntimeError(f"Model weights not found. Expected at: {model_path}")

    _model = build_unet()
    _model.load_state_dict(torch.load(model_path, map_location=_device))
    _model.to(_device)
    _model.eval()

    print(f"Model loaded from {model_path} on {_device}")
    return _model, _device


# --- response schema ---
class PredictionResponse(BaseModel):
    panel_coverage_percent: float
    mask_base64: str
    inference_time_ms: float
    image_size: tuple[int, int]


# --- preprocessing ---
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to model-ready tensor."""
    image = image.convert("RGB")
    image = image.resize((400, 400))
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


# --- endpoints ---
@app.get("/")
def root():
    return {"message": "Solar Panel Segmentation API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Segment solar panels in an aerial image.

    Upload a PNG or JPEG aerial image. Returns:
    - panel coverage percentage
    - predicted mask as base64 PNG
    - inference time in milliseconds
    """
    # validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG and PNG accepted."
        )

    # load image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    original_size = image.size

    # load model
    try:
        model, device = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # inference
    start_time = time.time()

    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    inference_time_ms = (time.time() - start_time) * 1000

    # compute panel coverage
    panel_pixels = mask.sum()
    total_pixels = mask.size
    coverage_percent = round((panel_pixels / total_pixels) * 100, 2)

    # encode mask as base64 PNG for response
    mask_image = Image.fromarray(mask * 255)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    mask_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return PredictionResponse(
        panel_coverage_percent=coverage_percent,
        mask_base64=mask_b64,
        inference_time_ms=round(inference_time_ms, 2),
        image_size=original_size,
    )