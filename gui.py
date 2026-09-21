#!/usr/bin/env python3
"""
Local 4K Upscaler Web GUI (ReelFrame)
FastAPI-powered modern web dashboard with real-time progress and GPU metrics.

Author: Raliq Hidayat BM3
Repository: https://github.com/kouji999/ReelFrame
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import core upscaling logic
from upscale import (
    IMAGE_EXTENSIONS,
    MODELS_DIR,
    SCRIPT_DIR,
    VIDEO_EXTENSIONS,
    resolve_model_path,
    upscale_image,
    upscale_video,
)
from spandrel import ModelLoader

app = FastAPI(title="ReelFrame - AI 4K Upscaler by Raliq Hidayat BM3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = SCRIPT_DIR / "temp_uploads"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory jobs tracker
JOBS: Dict[str, Dict[str, Any]] = {}

# Global model cache to avoid re-loading on each upscale
LOADED_MODELS: Dict[str, Any] = {}
MODEL_LOCK = threading.Lock()


def get_cached_model(model_name: str, device: torch.device):
    with MODEL_LOCK:
        if model_name in LOADED_MODELS:
            return LOADED_MODELS[model_name]
        
        model_file = resolve_model_path(model_name)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        desc = ModelLoader().load_from_file(str(model_file))
        model = desc.model.to(device).eval()
        if device.type == "cuda":
            model = model.half()
        LOADED_MODELS[model_name] = (model, desc.scale)
        return LOADED_MODELS[model_name]


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = SCRIPT_DIR / "web" / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/system-info")
async def get_system_info():
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "No NVIDIA GPU found"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        if cuda_avail
        else 0
    )
    
    models = []
    if MODELS_DIR.exists():
        for p in MODELS_DIR.glob("*.pth"):
            models.append(p.name)

    return {
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "models": models,
        "author": "Raliq Hidayat BM3"
    }


def run_upscale_job(job_id: str, input_path: Path, output_path: Path, model_name: str, scale_str: str, tile_size: int, cq: int):
    job = JOBS[job_id]
    job["status"] = "processing"
    job["log"] = f"Starting upscale job {job_id}...\n"

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_scale = get_cached_model(model_name, device)

        # Determine downscale factor:
        # If model is x4 and user asks for 2x, downscale = 0.5
        downscale = 0.5 if scale_str == "2x" else 1.0

        suffix = input_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            job["log"] += f"Processing image {input_path.name}...\n"
            upscale_image(
                src=input_path,
                dst=output_path,
                model=model,
                model_scale=model_scale,
                downscale=downscale,
                tile_size=tile_size,
                device=device,
            )
            job["progress"] = 100.0
            job["status"] = "completed"
            job["log"] += f"Successfully upscaled image to {output_path.name}!\n"

        elif suffix in VIDEO_EXTENSIONS:
            def on_progress(frame_i: int, total_frames: int, fps: float, eta: float):
                job["current_frame"] = frame_i
                job["total_frames"] = total_frames
                job["fps"] = fps
                job["eta"] = eta
                job["progress"] = round((frame_i / total_frames) * 100.0, 1) if total_frames else 0.0
                job["log"] = f"Processing frame {frame_i}/{total_frames} ({job['progress']}%) at {fps:.2f} fps. ETA: {round(eta)}s"

            upscale_video(
                src=input_path,
                dst=output_path,
                model=model,
                model_scale=model_scale,
                downscale=downscale,
                cq=cq,
                preset="p4",
                queue_size=8,
                tile_size=tile_size,
                encoder="hevc_nvenc",
                progress_callback=on_progress,
                device=device,
            )
            job["progress"] = 100.0
            job["status"] = "completed"
            job["log"] += f"\nVideo upscaling finished successfully -> {output_path.name}"
        else:
            raise ValueError(f"Unsupported format {suffix}")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["log"] += f"\n[ERROR] {e}"


@app.post("/api/upscale")
async def start_upscale_api(
    file: UploadFile = File(...),
    model: str = Form("realesr-general-x4v3.pth"),
    scale: str = Form("2x"),
    tile: int = Form(512),
    cq: int = Form(20),
):
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix
    saved_input = UPLOAD_DIR / f"{job_id}_input{ext}"
    output_filename = f"{Path(file.filename).stem}_4K{ext}"
    saved_output = OUTPUT_DIR / f"{job_id}_{output_filename}"

    with saved_input.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    JOBS[job_id] = {
        "job_id": job_id,
        "input_filename": file.filename,
        "output_filename": output_filename,
        "input_path": saved_input,
        "output_path": saved_output,
        "status": "queued",
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "fps": 0.0,
        "eta": 0.0,
        "log": "Uploaded and queued...",
        "error": None,
    }

    thread = threading.Thread(
        target=run_upscale_job,
        args=(job_id, saved_input, saved_output, model, scale, tile, cq),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "job_id": job_id}


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in JOBS:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return JOBS[job_id]


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    if job_id not in JOBS:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    job = JOBS[job_id]
    if job["status"] != "completed" or not job["output_path"].exists():
        return JSONResponse(status_code=400, content={"error": "Result file not ready"})
    return FileResponse(
        path=job["output_path"],
        filename=job["output_filename"],
        media_type="application/octet-stream",
    )


def main():
    host = "127.0.0.1"
    port = 7860
    url = f"http://{host}:{port}"
    print("=" * 60)
    print(" Local 4K Upscaler — Web Interface")
    print(" Author: Raliq Hidayat BM3")
    print(f" URL: {url}")
    print("=" * 60)

    # Open browser automatically after 1 second
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
