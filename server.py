"""FastAPI backend for the Attendance AI system."""

import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from attendance_ai.main import process_attendance_sheet

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──
app = FastAPI(
    title="Attendance AI",
    description="AI-powered attendance sheet recognition from photos",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ──
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
DEBUG_DIR = BASE_DIR / "debug_output"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# Serve debug images
app.mount("/debug", StaticFiles(directory=str(DEBUG_DIR)), name="debug")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
UPLOAD_TTL = 3600  # Keep uploaded files for 1 hour


def _cleanup_old_uploads():
    """Remove uploaded files older than UPLOAD_TTL seconds."""
    now = time.time()
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and now - f.stat().st_mtime > UPLOAD_TTL:
            f.unlink(missing_ok=True)


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/process")
async def process_image(file: UploadFile = File(...)):
    """Process an uploaded attendance sheet image.

    Returns JSON with student attendance data and summary.
    """
    # Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Save to temp file
    job_id = uuid.uuid4().hex[:12]
    safe_filename = f"{job_id}{ext}"
    upload_path = UPLOAD_DIR / safe_filename
    upload_path.write_bytes(content)

    job_debug_dir = str(DEBUG_DIR / job_id)

    try:
        logger.info(f"Processing job {job_id}: {file.filename} ({len(content)} bytes)")
        _cleanup_old_uploads()

        result = process_attendance_sheet(
            image_path=str(upload_path),
            debug=True,
            debug_dir=job_debug_dir,
        )

        # Convert debug image paths to URLs
        debug_urls = []
        for img_path in result.get("debug_images", []):
            rel = Path(img_path).relative_to(DEBUG_DIR)
            debug_urls.append(f"/debug/{rel.as_posix()}")

        response = {
            "job_id": job_id,
            "filename": file.filename,
            "students": result["students"],
            "student_info": result.get("student_info", []),
            "summary": result["summary"],
            "csv": result["csv_string"],
            "debug_images": debug_urls,
            "timing": result.get("timing", {}),
            "total_columns": result.get("total_columns", 0),
            "selected_column": result.get("selected_column", 0),
            "grid": result.get("grid", []),
        }

        if "error" in result:
            response["warning"] = result["error"]

        return JSONResponse(content=response)

    except Exception as e:
        logger.exception(f"Processing failed for job {job_id}")
        # Clean up upload on failure
        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "attendance-ai"}


@app.post("/api/reprocess")
async def reprocess_image(
    job_id: str = Form(...),
    column_index: int = Form(...),
):
    """Re-process a previously uploaded image with a different column.

    Expects job_id from a prior /api/process response and
    the desired column_index (0-based).
    """
    # Find the uploaded file by job_id prefix
    matches = list(UPLOAD_DIR.glob(f"{job_id}.*"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail="Image not found. Please upload the image again.",
        )

    upload_path = matches[0]
    job_debug_dir = str(DEBUG_DIR / job_id)

    try:
        logger.info(f"Reprocessing job {job_id} with column_index={column_index}")

        result = process_attendance_sheet(
            image_path=str(upload_path),
            debug=True,
            debug_dir=job_debug_dir,
            column_index=column_index,
        )

        debug_urls = []
        for img_path in result.get("debug_images", []):
            rel = Path(img_path).relative_to(DEBUG_DIR)
            debug_urls.append(f"/debug/{rel.as_posix()}")

        response = {
            "job_id": job_id,
            "students": result["students"],
            "student_info": result.get("student_info", []),
            "summary": result["summary"],
            "csv": result["csv_string"],
            "debug_images": debug_urls,
            "timing": result.get("timing", {}),
            "total_columns": result.get("total_columns", 0),
            "selected_column": result.get("selected_column", 0),
            "grid": result.get("grid", []),
        }

        if "error" in result:
            response["warning"] = result["error"]

        return JSONResponse(content=response)

    except Exception as e:
        logger.exception(f"Reprocessing failed for job {job_id}")
        raise HTTPException(status_code=500, detail=f"Reprocessing failed: {str(e)}")
