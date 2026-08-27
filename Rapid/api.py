from __future__ import annotations

import os
import re
import logging
import tempfile
import time

import fitz
import numpy as np
from rapidocr import RapidOCR

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse


# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI(
    title="RapidOCR Whole Document API",
    version="1.0.0",
)


# ==========================================================
# CONFIG
# ==========================================================

DPI = 250

OUTPUT_FOLDER = "OutputTXT"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==========================================================
# RAPIDOCR LOGGING
# ==========================================================

logging.getLogger(
    "rapidocr"
).setLevel(
    logging.ERROR
)


# ==========================================================
# RAPIDOCR ENGINE
# ==========================================================

engine = RapidOCR()


# ==========================================================
# RAPIDOCR - ONE IMAGE
# ==========================================================

def extract_text_from_image(image):

    result = engine(image)

    if (
        result is None
        or result.txts is None
    ):
        return ""

    return "\n".join(
        result.txts
    )


# ==========================================================
# OCR WHOLE PDF
# ==========================================================

def extract_text_from_pdf(pdf_path):

    document = fitz.open(
        pdf_path
    )

    total_pages = len(
        document
    )

    print()
    print("=" * 70)
    print("RAPIDOCR - WHOLE DOCUMENT OCR")
    print("=" * 70)
    print(
        f"PDF          : {pdf_path}"
    )
    print(
        f"Total pages  : {total_pages}"
    )
    print(
        f"DPI          : {DPI}"
    )
    print("=" * 70)
    print()

    # ------------------------------------------------------
    # Keep page number attached to OCR result
    # ------------------------------------------------------

    page_text = {}

    # ------------------------------------------------------
    # OCR EVERY PAGE
    # ------------------------------------------------------

    for page_number in range(
        total_pages
    ):

        print(
            f"Processing page "
            f"{page_number + 1}/{total_pages}..."
        )

        # --------------------------------------------------
        # Render page
        # --------------------------------------------------

        page = document[
            page_number
        ]

        pix = page.get_pixmap(
            dpi=DPI,
            alpha=False
        )

        # --------------------------------------------------
        # Convert to NumPy image
        # --------------------------------------------------

        image = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(
            pix.height,
            pix.width,
            3
        )

        # --------------------------------------------------
        # RapidOCR
        # --------------------------------------------------

        text = extract_text_from_image(
            image
        )

        # --------------------------------------------------
        # Preserve page order
        # --------------------------------------------------

        page_text[
            page_number + 1
        ] = text

    document.close()

    # ------------------------------------------------------
    # Combine pages IN ORDER
    # ------------------------------------------------------

    full_text = []

    for page_number in range(
        1,
        total_pages + 1
    ):

        full_text.append(
            page_text[
                page_number
            ]
        )

    combined_text = "\n\n".join(
        full_text
    )

    return (
        combined_text,
        total_pages
    )


# ==========================================================
# CLEAN COMPLETE OCR TEXT
# ==========================================================

def clean_text(text):

    if not text:
        return ""

    # ------------------------------------------------------
    # Normalize line endings
    # ------------------------------------------------------

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # ------------------------------------------------------
    # Remove trailing spaces
    # ------------------------------------------------------

    lines = []

    for line in text.split(
        "\n"
    ):

        line = line.strip()

        lines.append(
            line
        )

    text = "\n".join(
        lines
    )

    # ------------------------------------------------------
    # Remove excessive spaces
    # ------------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # ------------------------------------------------------
    # Remove excessive blank lines
    # ------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # ------------------------------------------------------
    # Remove obvious OCR garbage
    # ------------------------------------------------------

    text = re.sub(
        r"[|¦]{2,}",
        "",
        text
    )

    text = re.sub(
        r"_{3,}",
        "",
        text
    )

    # ------------------------------------------------------
    # Fix spaces before punctuation
    # ------------------------------------------------------

    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text
    )

    # ------------------------------------------------------
    # Fix missing spaces after punctuation
    # ------------------------------------------------------

    text = re.sub(
        r"([,.;:!?])(?=[A-Za-z])",
        r"\1 ",
        text
    )

    # ------------------------------------------------------
    # Remove spaces around brackets
    # ------------------------------------------------------

    text = re.sub(
        r"\(\s+",
        "(",
        text
    )

    text = re.sub(
        r"\s+\)",
        ")",
        text
    )

    # ------------------------------------------------------
    # Remove spaces around slashes
    # ------------------------------------------------------

    text = re.sub(
        r"\s*/\s*",
        "/",
        text
    )

    # ------------------------------------------------------
    # Final cleanup
    # ------------------------------------------------------

    return text.strip()


# ==========================================================
# SAVE TXT
# ==========================================================

def save_text(
    text,
    filename
):

    # ------------------------------------------------------
    # Get original filename without extension
    # ------------------------------------------------------

    stem = os.path.splitext(
        filename
    )[0]

    # ------------------------------------------------------
    # Save using uploaded PDF name
    # ------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"{stem}.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            text
        )

    return output_path


# ==========================================================
# HEALTH
# ==========================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "RapidOCR Whole Document API",
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
    file: UploadFile = File(...)
):

    # ------------------------------------------------------
    # Validate file
    # ------------------------------------------------------

    filename = (
        file.filename
        or "document.pdf"
    )

    if not filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # ------------------------------------------------------
    # Read uploaded PDF
    # ------------------------------------------------------

    try:

        pdf_bytes = await file.read()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read PDF: {exc}"
        )

    if not pdf_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty."
        )

    # ------------------------------------------------------
    # Create temporary PDF
    # ------------------------------------------------------

    temp_path = None

    start_time = time.perf_counter()

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp_file:

            temp_file.write(
                pdf_bytes
            )

            temp_path = (
                temp_file.name
            )

        # --------------------------------------------------
        # OCR WHOLE DOCUMENT
        # --------------------------------------------------

        raw_text, total_pages = (
            extract_text_from_pdf(
                temp_path
            )
        )

        print()
        print("-" * 70)
        print("ALL PAGES OCR COMPLETE")
        print("-" * 70)
        print(
            f"Pages OCR'd   : {total_pages}"
        )
        print(
            f"Raw characters: "
            f"{len(raw_text):,}"
        )

        # --------------------------------------------------
        # CLEAN AFTER OCR
        # --------------------------------------------------

        print()
        print(
            "Cleaning complete OCR text..."
        )

        cleaned_text = clean_text(
            raw_text
        )

        print(
            f"Clean characters: "
            f"{len(cleaned_text):,}"
        )

        # --------------------------------------------------
        # SAVE TXT
        # --------------------------------------------------

        output_path = save_text(
            cleaned_text,
            filename
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print()
        print("=" * 70)
        print("OCR COMPLETED")
        print("=" * 70)
        print(
            f"Pages processed : "
            f"{total_pages}"
        )
        print(
            f"Output          : "
            f"{output_path}"
        )
        print(
            f"Time            : "
            f"{elapsed:.2f} seconds"
        )
        print("=" * 70)

        # --------------------------------------------------
        # API RESPONSE
        # --------------------------------------------------

        return JSONResponse(
            content={
                "success": True,
                "filename": filename,
                "pages": total_pages,
                "characters": len(
                    cleaned_text
                ),
                "processing_time_seconds": round(
                    elapsed,
                    2
                ),
                "output_file": output_path,
                "text": cleaned_text,
            }
        )

    except Exception as exc:

        print()
        print(
            f"OCR ERROR: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {exc}"
        )

    finally:

        # --------------------------------------------------
        # Delete temporary PDF
        # --------------------------------------------------

        if temp_path and os.path.exists(
            temp_path
        ):

            try:

                os.remove(
                    temp_path
                )

            except OSError:

                pass


# ==========================================================
# RUN SERVER
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )