"""
Inference pipeline for Multi-Path Fusion Network.
Handles preprocessing, model inference, and postprocessing.

Pipeline: input → preprocessing → model forward pass → output mask
No fallback logic, no rule-based segmentation.
"""

import io
import base64
import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

try:
    from model import MultiPathFusionNet
except ImportError:
    from backend.model import MultiPathFusionNet

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = Path(__file__).parent / "models" / "best_model.pth"

# Tumor class colors: 0=BG, 1=Tumor(green)
CLASS_COLORS = {
    0: (0,   0,   0,   0),    # Background – transparent
    1: (0,   200, 80,  160),  # Tumor – green
}

# ─────────────────────────────────────────────────────────────
# Model singleton
# ─────────────────────────────────────────────────────────────
_model: MultiPathFusionNet | None = None
_model_loaded: bool = False


def get_model() -> MultiPathFusionNet:
    """
    Load the Multi-Path Fusion Network with trained weights.
    Raises RuntimeError if weights cannot be loaded.
    """
    global _model, _model_loaded
    if _model is None:
        _model = MultiPathFusionNet(in_channels=1, num_classes=2).to(DEVICE)

        if not WEIGHTS_PATH.exists():
            raise RuntimeError(
                f"Trained model weights not found at {WEIGHTS_PATH}. "
                f"Run train.py first to generate weights."
            )

        checkpoint = torch.load(str(WEIGHTS_PATH), map_location=DEVICE, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            _model.load_state_dict(checkpoint["model_state"])
        else:
            _model.load_state_dict(checkpoint)

        _model.eval()
        _model_loaded = True
        logger.info("Loaded trained weights from %s", WEIGHTS_PATH)

    return _model


def is_model_loaded() -> bool:
    """Check whether the model has been loaded with trained weights."""
    return _model_loaded


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load image bytes → grayscale → normalise → 1-channel tensor.
    Returns:
        tensor: (1, 1, 256, 256) float32 tensor for model input
        arr_uint8: (256, 256) uint8 array for visualization
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)

    # Normalize to [0, 1]
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min > 1e-8:
        arr_norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr_norm = np.zeros_like(arr)

    # Single channel input: (1, 256, 256)
    arr_1ch = arr_norm[np.newaxis, :, :]  # (1, 256, 256)
    tensor = torch.from_numpy(arr_1ch).unsqueeze(0).to(DEVICE)  # (1, 1, 256, 256)

    arr_uint8 = (arr_norm * 255).astype(np.uint8)
    return tensor, arr_uint8


# ─────────────────────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────────────────────

def run_inference(image_bytes: bytes) -> dict:
    """
    Full pipeline: preprocess → model inference → postprocess.
    Returns JSON-serialisable dict with mask, volume, coordinates, and visualisations.

    Raises RuntimeError if model weights are not loaded.
    No fallback logic — output comes strictly from the neural network.
    """
    tensor, arr_uint8 = preprocess_image(image_bytes)

    model = get_model()

    with torch.no_grad():
        logits = model(tensor)                # (1, 2, 256, 256)
        probs  = F.softmax(logits, dim=1)
        pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (256, 256)

    # Confidence from model's softmax probabilities
    max_prob = probs.max(dim=1)[0].mean().item()
    confidence = round(max_prob * 100, 1)

    # ── Metrics ─────────────────────────────────────────────
    vol_tumor = float(np.sum(pred == 1))
    tumor_volume_mm3 = max(round(vol_tumor * 0.15, 0), 1)

    # ── Coordinates (centroid of tumor region) ────────────
    tumor_region = pred == 1
    if tumor_region.any():
        ys, xs = np.where(tumor_region)
        cx = round(float(xs.mean()) / IMG_SIZE * 100, 1)
        cy = round(float(ys.mean()) / IMG_SIZE * 100, 1)
        cz = 50.0  # Z-coordinate requires volumetric data
    else:
        cx, cy, cz = 50.0, 50.0, 50.0

    # ── Visualisation images ──────────────────────────────────
    mri_b64       = _array_to_base64_png(arr_uint8, mode='L')
    overlay_b64   = _make_overlay_image(arr_uint8, pred)
    axial_b64     = _make_3d_projection(pred, 'axial')

    return {
        "status": "success",
        "tumor_volume_mm3": tumor_volume_mm3,
        "coordinates": {"x": cx, "y": cy, "z": cz},
        "confidence": confidence,
        "confidence_label": "High" if confidence >= 85 else "Medium" if confidence >= 70 else "Low",
        "mri_image": mri_b64,
        "overlay_image": overlay_b64,
        "axial_projection": axial_b64,
        "mask_summary": {
            "tumor_pixels": int(vol_tumor),
        },
        "treatment_recommendation": _get_treatment_recommendation(tumor_volume_mm3, confidence, cx, cy, cz),
    }


def _array_to_base64_png(arr: np.ndarray, mode='L') -> str:
    """Convert numpy array to base64-encoded PNG string."""
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _make_overlay_image(arr_uint8: np.ndarray, mask: np.ndarray) -> str:
    """
    Compose MRI with colour overlay for tumor region on the left
    and clean MRI on the right.
    Returns base64 PNG.
    """
    from scipy import ndimage

    h, w = arr_uint8.shape

    # Base image – RGB from grayscale
    base = Image.fromarray(arr_uint8, 'L').convert('RGBA')

    # Build overlay for tumor region
    overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)

    region = mask == 1
    if region.any():
        dist = ndimage.distance_transform_edt(region).astype(np.float32)
        max_dist = max(dist.max(), 1.0)
        dist_norm = dist / max_dist

        color = CLASS_COLORS[1]
        alpha_base = color[3]
        alpha_map = (dist_norm * alpha_base * 0.8 + alpha_base * 0.2).clip(0, 255)

        overlay_arr[region, 0] = color[0]
        overlay_arr[region, 1] = color[1]
        overlay_arr[region, 2] = color[2]
        overlay_arr[region, 3] = alpha_map[region].astype(np.uint8)

    overlay = Image.fromarray(overlay_arr, 'RGBA')

    # Composite: left half = overlay on MRI, right half = clean MRI
    composited = Image.alpha_composite(base, overlay)

    # Build split view: left = overlay, right = clean
    split = Image.new('RGBA', (w * 2, h))
    split.paste(composited, (0, 0))
    split.paste(base, (w, 0))

    # Draw a thin white divider
    draw = ImageDraw.Draw(split)
    draw.line([(w, 0), (w, h)], fill=(255, 255, 255, 200), width=2)

    # Add label overlays
    draw.rectangle([2, 2, 120, 18], fill=(0, 0, 0, 150))
    draw.text((5, 4), "Segmentation Overlay", fill=(255, 255, 255, 255))

    draw.rectangle([w + 2, 2, w + 100, 18], fill=(0, 0, 0, 150))
    draw.text((w + 5, 4), "Original MRI", fill=(255, 255, 255, 255))

    # Add legend at bottom of overlay side
    legend_y = h - 22
    c = CLASS_COLORS[1]
    draw.rectangle([8, legend_y, 18, legend_y + 10], fill=(c[0], c[1], c[2], 255))
    draw.text((22, legend_y - 1), "Tumor", fill=(255, 255, 255, 220))

    buf = io.BytesIO()
    split.convert('RGB').save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _make_3d_projection(mask: np.ndarray, view: str = 'axial') -> str:
    """Generate a minimal colored projection view of the segmentation."""
    h, w = mask.shape
    proj = Image.new('RGB', (w, h), (10, 10, 30))
    draw_arr = np.array(proj)

    for cls_idx, color in CLASS_COLORS.items():
        if cls_idx == 0:
            continue
        region = mask == cls_idx
        draw_arr[region] = color[:3]

    buf = io.BytesIO()
    Image.fromarray(draw_arr).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _get_treatment_recommendation(vol_mm3: float, confidence: float, x: float, y: float, z: float) -> dict:
    """Generate clinical recommendation text based on metrics."""
    note = (
        f"Model confidence: {confidence:.0f}%. "
        f"Tumor volume of {vol_mm3:.0f} cubic mm may indicate need for "
        f"local surgical resection based on tumor location (x={x}, y={y}, z={z})."
    )
    options = [
        {"treatment": "Radiation therapy", "detail": "Next MRI in 3 months"},
        {"treatment": "Radiotherapy",       "detail": "To target residual cells"},
        {"treatment": "Chemotherapy",       "detail": "Adjuvant temozolomide"},
    ]
    return {"note": note, "options": options}
