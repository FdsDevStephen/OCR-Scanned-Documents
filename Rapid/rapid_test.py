import os
import re
import logging

import fitz
import numpy as np
from rapidocr import RapidOCR


# ==========================================================
# CONFIG
# ==========================================================

PDF_PATH = r"C:\Users\steph\OneDrive\Desktop\OCR\WP-12860-2025-B.pdf"

DPI = 250

OUTPUT_FOLDER = "OutputTXT"
OUTPUT_FILE = "output.txt"


# ==========================================================
# RAPIDOCR LOGGING
# ==========================================================

logging.getLogger("rapidocr").setLevel(logging.ERROR)


# ==========================================================
# RAPIDOCR ENGINE
# ==========================================================

engine = RapidOCR()


# ==========================================================
# RAPIDOCR - ONE IMAGE
# ==========================================================

def extract_text_from_image(image):

    result = engine(image)

    if result is None or result.txts is None:
        return ""

    return "\n".join(result.txts)


# ==========================================================
# STEP 1-5
#
# Open PDF
# Get total pages
# Render every page
# RapidOCR every page
# Preserve page order
# ==========================================================

def extract_text_from_pdf(pdf_path):

    document = fitz.open(pdf_path)

    total_pages = len(document)

    print()
    print("=" * 70)
    print("RAPIDOCR - WHOLE DOCUMENT OCR")
    print("=" * 70)
    print(f"PDF          : {pdf_path}")
    print(f"Total pages  : {total_pages}")
    print(f"DPI          : {DPI}")
    print("=" * 70)
    print()

    # Dictionary keeps the page number attached
    # to its OCR result.
    page_text = {}

    for page_number in range(total_pages):

        print(
            f"Processing page "
            f"{page_number + 1}/{total_pages}..."
        )

        # --------------------------------------------------
        # Render page
        # --------------------------------------------------

        page = document[page_number]

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

        page_text[page_number + 1] = text

    document.close()

    # ------------------------------------------------------
    # Combine pages IN ORDER
    # ------------------------------------------------------

    full_text = []

    for page_number in range(
        1,
        total_pages + 1
    ):

        text = page_text[
            page_number
        ]

        full_text.append(text)

    combined_text = "\n\n".join(
        full_text
    )

    return combined_text, total_pages


# ==========================================================
# STEP 6
#
# CLEAN THE COMBINED OCR TEXT
#
# IMPORTANT:
# Cleaning happens AFTER the entire document
# has been OCR'd.
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

    for line in text.split("\n"):

        line = line.strip()

        lines.append(line)

    text = "\n".join(lines)

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
        r"[_]{3,}",
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

    text = text.strip()

    return text


# ==========================================================
# STEP 7
#
# SAVE ONE TXT FILE
# ==========================================================

def save_text(
    text,
    output_folder,
    output_file
):

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    output_path = os.path.join(
        output_folder,
        output_file
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(text)

    return output_path


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # OCR COMPLETE DOCUMENT
    # ------------------------------------------------------

    raw_text, total_pages = (
        extract_text_from_pdf(
            PDF_PATH
        )
    )

    print()
    print("-" * 70)
    print("ALL PAGES OCR COMPLETE")
    print("-" * 70)
    print(
        f"Pages OCR'd: {total_pages}"
    )
    print(
        f"Raw characters: {len(raw_text):,}"
    )

    # ------------------------------------------------------
    # CLEAN AFTER OCR
    # ------------------------------------------------------

    print()
    print("Cleaning complete OCR text...")

    cleaned_text = clean_text(
        raw_text
    )

    print(
        f"Clean characters: "
        f"{len(cleaned_text):,}"
    )

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    output_path = save_text(
        cleaned_text,
        OUTPUT_FOLDER,
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print("OCR COMPLETED")
    print("=" * 70)
    print(
        f"Pages processed : {total_pages}"
    )
    print(
        f"Output          : {output_path}"
    )
    print("=" * 70)