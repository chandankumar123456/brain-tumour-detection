"""
FastAPI Backend – Brain Tumor Segmentation Dashboard
Endpoints: upload MRI for real inference, health check, generate report.
"""

import io
import json
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path

try:
    from inference import run_inference, is_model_loaded
    from report_generator import generate_pdf_report
except ImportError:
    from backend.inference import run_inference, is_model_loaded
    from backend.report_generator import generate_pdf_report

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="Multi-Path Fusion Network with Global Attention (Wu et al. 2023)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the React frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def root():
    """Serve the dashboard."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Brain Tumor Segmentation API running. Visit /docs for API docs."})


@app.get("/api/health")
async def health():
    """Health check endpoint with real model status."""
    weights_path = Path(__file__).parent / "models" / "best_model.pth"
    return {
        "status": "ok",
        "model": "MultiPathFusionNet",
        "version": "1.0.0",
        "weights_loaded": is_model_loaded(),
        "weights_available": weights_path.exists(),
    }


@app.post("/api/upload")
async def upload_mri(file: UploadFile = File(...)):
    """
    Upload an MRI image (JPG/PNG) for brain tumor segmentation.
    Returns segmentation mask, volume, coordinates, and visualisations.
    All outputs come strictly from the trained neural network.
    """
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        result = run_inference(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")

    return JSONResponse(result)


@app.post("/api/report/pdf")
async def download_pdf_report(result: dict):
    """Generate and download PDF report from analysis result."""
    try:
        pdf_bytes = generate_pdf_report(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(exc)}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=brain_tumor_report.pdf"},
    )


@app.post("/api/report/json")
async def export_json_report(result: dict):
    """Export analysis result as downloadable JSON."""
    return Response(
        content=json.dumps(result, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=brain_tumor_report.json"},
    )
