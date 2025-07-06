from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from models import warmup_models
from loguru import logger
import uvicorn
from pydantic import BaseModel
import numpy as np
import tempfile
import shutil
import requests
from utils.frame_extractor import extract_frames
import base64
import matplotlib.pyplot as plt
import io
import time
import psutil
import asyncio
try:
    import torch
except ImportError:
    torch = None
from typing import Optional, Dict
import cv2

app = FastAPI(title="Deepfake Detection API", description="Production-ready deepfake detection backend.", version="1.0.0")

logger.info("Warming up models...")
models = warmup_models()
logger.info("Models loaded: {}", list(models.keys()))

# Default weights for ensemble
WEIGHTS = {
    'facexray': 0.25,
    'efficientnet': 0.25,
    'f3net': 0.25,
    'lipforensics': 0.25
}

# Metrics storage
METRICS = {
    'total_requests': 0,
    'total_processing_time': 0.0,
    'total_frames': 0,
    'last_processing_time': 0.0,
    'last_frames': 0,
}

# Request queueing and memory management
MAX_CONCURRENT_REQUESTS = 2
MIN_FREE_MEMORY_MB = 100  # Lowered from 1000 to 100
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def has_enough_memory():
    mem = psutil.virtual_memory()
    free_mb = mem.available / 1e6
    return free_mb > MIN_FREE_MEMORY_MB

def update_metrics(processing_time, frames):
    METRICS['total_requests'] += 1
    METRICS['total_processing_time'] += processing_time
    METRICS['total_frames'] += frames
    METRICS['last_processing_time'] = processing_time
    METRICS['last_frames'] = frames

class ModelBreakdown(BaseModel):
    facexray: float
    efficientnet: float
    f3net: float
    lipforensics: float

class PredictionResult(BaseModel):
    is_fake: bool
    confidence: float
    model_breakdown: ModelBreakdown

class MetricsResult(BaseModel):
    processing_time: str
    frames_analyzed: int

class PredictionResponse(BaseModel):
    prediction: PredictionResult
    metrics: MetricsResult
    visualization: str

class AnalyzeUrlRequest(BaseModel):
    url: str

class WeightsConfig(BaseModel):
    facexray: Optional[float] = 0.25
    efficientnet: Optional[float] = 0.25
    f3net: Optional[float] = 0.25
    lipforensics: Optional[float] = 0.25

class MetricsResponse(BaseModel):
    total_requests: int
    average_processing_time: float
    average_frames_analyzed: float
    last_processing_time: float
    last_frames_analyzed: int
    system: Dict[str, float]

def run_ensemble(frames, models, weights=WEIGHTS):
    breakdown = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for name, model in models.items():
        score = float(model.predict(frames))
        breakdown[name] = score
        w = weights.get(name, 0.0)
        weighted_sum += score * w
        total_weight += w
    confidence = float(weighted_sum / total_weight) if total_weight > 0 else 0.0
    is_fake = confidence > 0.5
    return is_fake, confidence, breakdown

def plot_results(breakdown):
    # Remove bar chart visualization, just return None
    return None

@app.get("/")
def root():
    return {"message": "Deepfake Detection API is running."}

@app.post(
    "/analyze/file",
    response_model=PredictionResponse,
    summary="Analyze uploaded file for deepfakes",
    description="Upload an image or video file for deepfake analysis using an ensemble of models."
)
async def analyze_file(
    file: UploadFile = File(..., description="Image or video file to analyze")
):
    """Analyze an uploaded file for deepfakes."""
    async with semaphore:
        if not has_enough_memory():
            logger.warning("Insufficient memory for request.")
            raise HTTPException(status_code=503, detail="Server busy: insufficient memory.")
        start = time.time()
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
            frames = extract_frames(tmp_path, fps=1, max_frames=32)
            if not frames:
                raise HTTPException(status_code=400, detail="No frames extracted.")
            is_fake, confidence, breakdown = run_ensemble(frames, models)
            processing_time = time.time() - start
            # Get the first frame as a thumbnail
            _, img_encoded = cv2.imencode('.jpg', frames[0])
            thumbnail = base64.b64encode(img_encoded).decode('utf-8')
            update_metrics(processing_time, len(frames))
            return JSONResponse({
                "prediction": {
                    "is_fake": is_fake,
                    "confidence": confidence,
                    "model_breakdown": breakdown
                },
                "metrics": {
                    "processing_time": f"{processing_time:.2f}s",
                    "frames_analyzed": len(frames)
                },
                "visualization": None,
                "thumbnail": thumbnail
            })
        except Exception as e:
            logger.exception("Error in /analyze/file")
            raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/analyze/url",
    response_model=PredictionResponse,
    summary="Analyze file from URL for deepfakes",
    description="Provide a URL to an image or video for deepfake analysis using an ensemble of models."
)
async def analyze_url(
    request: AnalyzeUrlRequest
):
    """Analyze a file from a URL for deepfakes."""
    url = request.url
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    if not url.lower().endswith(VIDEO_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Please provide a direct video file URL (ending in .mp4, .avi, .mov, etc.), not a website URL."
        )
    async with semaphore:
        if not has_enough_memory():
            logger.warning("Insufficient memory for request.")
            raise HTTPException(status_code=503, detail="Server busy: insufficient memory.")
        start = time.time()
        try:
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download file from URL.")
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name
            frames = extract_frames(tmp_path, fps=1, max_frames=32)
            if not frames:
                raise HTTPException(status_code=400, detail="No frames extracted.")
            is_fake, confidence, breakdown = run_ensemble(frames, models)
            processing_time = time.time() - start
            # Get the first frame as a thumbnail
            _, img_encoded = cv2.imencode('.jpg', frames[0])
            thumbnail = base64.b64encode(img_encoded).decode('utf-8')
            update_metrics(processing_time, len(frames))
            return JSONResponse({
                "prediction": {
                    "is_fake": is_fake,
                    "confidence": confidence,
                    "model_breakdown": breakdown
                },
                "metrics": {
                    "processing_time": f"{processing_time:.2f}s",
                    "frames_analyzed": len(frames)
                },
                "visualization": None,
                "thumbnail": thumbnail
            })
        except Exception as e:
            logger.exception("Error in /analyze/url")
            raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get model and system performance metrics",
    description="Returns aggregate and recent performance statistics, including processing time, frames analyzed, and system resource usage."
)
def get_metrics():
    """Get model and system performance metrics."""
    avg_time = METRICS['total_processing_time'] / METRICS['total_requests'] if METRICS['total_requests'] else 0.0
    avg_frames = METRICS['total_frames'] / METRICS['total_requests'] if METRICS['total_requests'] else 0.0
    sys_stats = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
    }
    if torch and torch.cuda.is_available():
        sys_stats['gpu_memory_allocated_MB'] = torch.cuda.memory_allocated() / 1e6
        sys_stats['gpu_memory_reserved_MB'] = torch.cuda.memory_reserved() / 1e6
    return {
        'total_requests': METRICS['total_requests'],
        'average_processing_time': avg_time,
        'average_frames_analyzed': avg_frames,
        'last_processing_time': METRICS['last_processing_time'],
        'last_frames_analyzed': METRICS['last_frames'],
        'system': sys_stats
    }

@app.get(
    "/config",
    response_model=WeightsConfig,
    summary="Get ensemble model weights",
    description="Returns the current weights used for the ensemble voting system."
)
def get_weights():
    """Get current ensemble weights."""
    return {"weights": WEIGHTS}

@app.post(
    "/config",
    response_model=WeightsConfig,
    summary="Set ensemble model weights",
    description="Set the weights for the ensemble voting system. Weights should sum to 1.0 for best results."
)
def set_weights(new_weights: WeightsConfig = Body(...)):
    """Set ensemble model weights."""
    for k in WEIGHTS:
        if k in new_weights:
            WEIGHTS[k] = float(new_weights[k])
    return {"weights": WEIGHTS}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 