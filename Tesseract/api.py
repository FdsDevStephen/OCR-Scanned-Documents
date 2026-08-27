from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ocr import OCRProcessor


# ==========================================================
# CONFIG
# ==========================================================

TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

OUTPUT_FOLDER = Path(
    os.getenv(
        "OCR_OUTPUT_FOLDER",
        "ocr_output",
    )
)

OCR_DPI = int(
    os.getenv(
        "OCR_DPI",
        "220",
    )
)

OCR_PSM = int(
    os.getenv(
        "OCR_PSM",
        "6",
    )
)

OCR_LANGUAGE = os.getenv(
    "OCR_LANGUAGE",
    "eng",
)

OCR_WORKERS = int(
    os.getenv(
        "OCR_WORKERS",
        "8",
    )
)


# ==========================================================
# FASTAPI APP
# ==========================================================

app = FastAPI(
    title="CCMS OCR API",
    description="Whole-document OCR API",
    version="1.0.0",
)


# ==========================================================
# OCR PROCESSOR
# ==========================================================

ocr_processor = OCRProcessor(
    tesseract_path=TESSERACT_PATH,
    output_folder=OUTPUT_FOLDER,
    dpi=OCR_DPI,
    psm=OCR_PSM,
    language=OCR_LANGUAGE,
    max_workers=OCR_WORKERS,
    denoise=True,
)


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "CCMS OCR API",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ==========================================================
# OCR ENDPOINT
# ==========================================================

@app.post("/ocr")
async def ocr_document(
    file: UploadFile = File(...),
):
    """
    Upload a PDF and OCR the ENTIRE document.

    Every page is processed.

    Returns the complete cleaned OCR text.
    """

    # ------------------------------------------------------
    # Validate filename
    # ------------------------------------------------------

    filename = file.filename or "document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # ------------------------------------------------------
    # Read uploaded PDF
    # ------------------------------------------------------

    try:
        pdf_bytes = await file.read()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded file: {exc}",
        )

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty.",
        )

    # ------------------------------------------------------
    # OCR
    # ------------------------------------------------------

    start = time.perf_counter()

    try:

        text = ocr_processor.process_bytes(
            pdf_bytes,
            filename,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {exc}",
        )

    elapsed = time.perf_counter() - start

    # ------------------------------------------------------
    # Response
    # ------------------------------------------------------

    return JSONResponse(
        content={
            "success": True,
            "filename": filename,
            "characters": len(text),
            "processing_time_seconds": round(
                elapsed,
                3,
            ),
            "text": text,
        }
    )


# ==========================================================
# RUN DIRECTLY
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )