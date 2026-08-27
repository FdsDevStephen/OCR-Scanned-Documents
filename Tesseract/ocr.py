from __future__ import annotations

import os
import re
import tempfile

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import fitz
import numpy as np
import pytesseract


# ==========================================================
# CONFIG
# ==========================================================

@dataclass(frozen=True)
class OCRConfig:
    """
    Configuration for whole-document OCR.

    Every page of the uploaded PDF is processed.
    There is no fast scan, page search, section detection,
    Prayer detection, or selective OCR.
    """

    # ----- OCR -----

    dpi: int = 220

    # Tesseract page segmentation mode.
    # 6 = Assume a single uniform block of text.
    psm: int = 6

    # Tesseract language.
    language: str = "eng"

    # ----- Preprocessing -----

    denoise: bool = True

    # ----- Concurrency -----

    max_workers: int = field(
        default_factory=lambda: min(8, os.cpu_count() or 4)
    )


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize(text: str) -> str:
    return " ".join(text.upper().split())


def clean_ocr_text(text: str) -> str:
    """
    Clean the COMPLETE OCR output.

    OCR flow:

        OCR ALL PAGES
             ↓
        combine all pages
             ↓
        clean_ocr_text()
             ↓
        final document text

    This does NOT summarize or paraphrase the document.
    """

    if not text:
        return ""

    # ======================================================
    # 1. NORMALIZE LINE ENDINGS
    # ======================================================

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []

    for raw_line in text.split("\n"):

        line = raw_line.strip()

        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        # ==================================================
        # 2. REMOVE TABLE / SCAN ARTIFACTS
        # ==================================================

        # Remove vertical table borders.
        line = re.sub(r"[|¦]", " ", line)

        # Remove long horizontal OCR lines.
        line = re.sub(r"[-_=]{4,}", " ", line)
        line = re.sub(r"_{3,}", " ", line)

        # Remove obvious decorative OCR symbols.
        line = re.sub(
            r"(?<!\w)[*@#~`^°¢£€©®•·]+(?!\w)",
            " ",
            line,
        )

        # Remove isolated backslashes.
        line = re.sub(r"(?<!\w)\\(?!\w)", " ", line)

        # ==================================================
        # 3. HIGH-CONFIDENCE OCR CORRECTIONS
        # ==================================================

        corrections = {
            # Court
            r"\bCORT\b": "COURT",
            r"\bCOUT\b": "COURT",
            r"\bCOORT\b": "COURT",

            # Karnataka
            r"\bKamataka\b": "Karnataka",
            r"\bKarnatake\b": "Karnataka",

            # Common OCR errors
            r"\bfram\b": "from",
            r"\bfrorm\b": "from",
            r"\bforma!\b": "formal",
            r"\brnarked\b": "marked",
            r"\bmace\b": "made",
            r"\bthrougn\b": "through",

            # Legal words
            r"\bpetitloner\b": "petitioner",
            r"\bPetitloner\b": "Petitioner",
            r"\bpetitlon\b": "petition",
            r"\brespondant\b": "respondent",

            r"\bGommissioner\b": "Commissioner",
            r"\bCommisioner\b": "Commissioner",
            r"\bCommissloner\b": "Commissioner",

            r"\bAsslstant\b": "Assistant",
            r"\bReyenue\b": "Revenue",

            r"\bGovemment\b": "Government",
            r"\bgoverment\b": "government",

            r"\bappllcation\b": "application",
            r"\bapproprlate\b": "appropriate",
            r"\bopportunlty\b": "opportunity",

            r"\bunauthorlzed\b": "unauthorized",
            r"\bregularizatlon\b": "regularization",
            r"\brepresentatlon\b": "representation",
            r"\bcancellatlon\b": "cancellation",

            r"\bproceedlng\b": "proceeding",
            r"\bproceedlngs\b": "proceedings",

            r"\bnotlce\b": "notice",

            r"\bOrignial\b": "Original",
            r"\bAmnexure\b": "Annexure",
            r"\bANNEXCURE\b": "ANNEXURE",

            r"\bAlfidavil\b": "Affidavit",
            r"\battidavit\b": "affidavit",
            r"\bVeritying\b": "Verifying",
            r"\bMemorand\b": "Memorandum",

            # Other recurring OCR errors
            r"\bLatter\b": "Letter",
            r"\bDio\b": "D/o",
            r"\bWio\b": "W/o",
        }

        for pattern, replacement in corrections.items():
            line = re.sub(
                pattern,
                replacement,
                line,
            )

        # ==================================================
        # 4. CONTEXT-SPECIFIC LEGAL CORRECTIONS
        # ==================================================

        # HIGH CORT OF KARNATAKA → HIGH COURT
        line = re.sub(
            r"\bHIGH\s+CORT\b",
            "HIGH COURT",
            line,
            flags=re.IGNORECASE,
        )

        # Farr/Far No.53 → Form No.53
        line = re.sub(
            r"\bFarr\s+No\.",
            "Form No.",
            line,
            flags=re.IGNORECASE,
        )

        line = re.sub(
            r"\bFar\s+No\.",
            "Form No.",
            line,
            flags=re.IGNORECASE,
        )

        # Rule 108 D (3)
        line = re.sub(
            r"\bRule\s+108\s+D\s*\(\s*3\s*\)",
            "Rule 108-D(3)",
            line,
            flags=re.IGNORECASE,
        )

        # Rule 108-D-3
        line = re.sub(
            r"\bRule\s+108-D-3\b",
            "Rule 108-D(3)",
            line,
            flags=re.IGNORECASE,
        )

        # ==================================================
        # 5. WHITESPACE
        # ==================================================

        line = re.sub(r"[ \t]+", " ", line)

        # Space before punctuation.
        line = re.sub(
            r"\s+([,.;:!?])",
            r"\1",
            line,
        )

        # Missing space after punctuation.
        line = re.sub(
            r"([,.;:!?])(?=[A-Za-z])",
            r"\1 ",
            line,
        )

        line = line.strip()

        if not line:
            continue

        # ==================================================
        # 6. REMOVE OBVIOUS GARBAGE LINES
        # ==================================================

        # Standalone page number.
        if re.fullmatch(r"\d{1,3}", line):
            continue

        # Standalone punctuation/symbols.
        if re.fullmatch(r"[\W_]+", line):
            continue

        # Tiny OCR garbage.
        if re.fullmatch(r"[A-Za-z]{1,2}", line):
            if line.upper() not in {
                "IN",
                "OF",
                "TO",
                "BY",
                "OR",
                "NO",
                "RS",
                "MR",
                "MS",
                "DR",
                "VS",
                "WP",
                "IA",
                "RA",
                "AND",
                "AS",
                "IS",
                "ON",
                "AT",
                "A",
                "I",
            }:
                continue

        lines.append(line)

    # ======================================================
    # 7. REBUILD BROKEN OCR LINES
    # ======================================================

    final_lines = []

    for line in lines:

        if not line:
            if final_lines and final_lines[-1] != "":
                final_lines.append("")
            continue

        # Never merge headings.
        is_heading = (
            line.isupper()
            and len(line) <= 100
        )

        # Never merge numbered legal paragraphs.
        is_numbered = bool(
            re.match(
                r"^(?:\d+[.)]|\([A-Za-z0-9]+\)|[A-Za-z][.)])\s+",
                line,
            )
        )

        if (
            final_lines
            and final_lines[-1]
            and not is_heading
            and not is_numbered
        ):

            previous = final_lines[-1]

            previous_is_heading = (
                previous.isupper()
                and len(previous) <= 100
            )

            if not previous_is_heading:

                # Join obvious wrapped sentences.
                if (
                    not previous.endswith(
                        (".", ":", ";", "?", "!")
                    )
                    and not line.startswith(
                        (
                            "ANNEXURE",
                            "INDEX",
                            "SYNOPSIS",
                            "WHEREFORE",
                            "BENGALURU",
                            "DATE:",
                            "ADVOCATE",
                            "BETWEEN:",
                            "AND:",
                        )
                    )
                ):
                    final_lines[-1] = (
                        previous.rstrip()
                        + " "
                        + line.lstrip()
                    )
                    continue

        final_lines.append(line)

    # ======================================================
    # 8. FINAL BLANK-LINE CLEANUP
    # ======================================================

    output = []

    for line in final_lines:

        if not line.strip():

            if output and output[-1] != "":
                output.append("")

        else:
            output.append(line.rstrip())

    while output and not output[0].strip():
        output.pop(0)

    while output and not output[-1].strip():
        output.pop()

    return "\n".join(output)


# ==========================================================
# PDF PAGE COUNT
# ==========================================================

def get_total_pages(pdf_path: Path) -> int:
    """
    Get the total number of pages in the PDF.
    """

    document = fitz.open(str(pdf_path))

    try:
        return document.page_count
    finally:
        document.close()


# ==========================================================
# RENDER PAGE
# ==========================================================

def render_page_gray(
    pdf_path: Path,
    page_number: int,
    dpi: int,
) -> np.ndarray:
    """
    Render ONE PDF page as grayscale.
    """

    document = fitz.open(str(pdf_path))

    try:

        pixmap = document[page_number - 1].get_pixmap(
            dpi=dpi,
            colorspace=fitz.csGRAY,
            alpha=False,
        )

        return np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        ).reshape(
            pixmap.height,
            pixmap.width,
        )

    finally:
        document.close()


# ==========================================================
# TESSERACT OCR
# ==========================================================

def ocr_gray(
    gray: np.ndarray,
    psm: int,
    dpi: int,
    language: str,
) -> str:
    """
    Run Tesseract on a grayscale page.
    """

    return pytesseract.image_to_string(
        gray,
        lang=language,
        config=(
            f"--oem 3 "
            f"--psm {psm} "
            f"-c user_defined_dpi={dpi}"
        ),
    )


# ==========================================================
# OCR ONE PAGE
# ==========================================================

def ocr_page(
    pdf_path: Path,
    page_number: int,
    cfg: OCRConfig,
) -> tuple[int, str]:
    """
    OCR exactly ONE page.

    Every page in the document comes through this function.
    """

    gray = render_page_gray(
        pdf_path,
        page_number,
        cfg.dpi,
    )

    if cfg.denoise:
        gray = cv2.medianBlur(gray, 3)

    text = ocr_gray(
        gray,
        cfg.psm,
        cfg.dpi,
        cfg.language,
    )

    return page_number, text


# ==========================================================
# OCR ALL PAGES
# ==========================================================

def ocr_document(
    pdf_path: Path,
    cfg: OCRConfig,
) -> dict[int, str]:
    """
    OCR EVERY PAGE of the document.

    There is no page selection.

    Example:

        50-page PDF
             ↓
        pages = [1, 2, 3, ..., 50]
             ↓
        OCR all 50 pages
    """

    total_pages = get_total_pages(pdf_path)

    print()
    print("=" * 70)
    print("WHOLE DOCUMENT OCR")
    print("=" * 70)
    print(f"PDF: {pdf_path.name}")
    print(f"Total pages: {total_pages}")
    print(f"DPI: {cfg.dpi}")
    print(f"PSM: {cfg.psm}")
    print(f"Workers: {cfg.max_workers}")
    print("=" * 70)
    print()

    if total_pages == 0:
        return {}

    pages = list(range(1, total_pages + 1))

    page_text: dict[int, str] = {}

    start = perf_counter()

    with ThreadPoolExecutor(
        max_workers=cfg.max_workers
    ) as executor:

        futures = {
            executor.submit(
                ocr_page,
                pdf_path,
                page,
                cfg,
            ): page
            for page in pages
        }

        completed = 0

        for future in as_completed(futures):

            page = futures[future]

            try:

                page_number, text = future.result()

                page_text[page_number] = text

                completed += 1

                print(
                    f"  OCR page {page_number}/{total_pages}"
                    f"  ({completed}/{total_pages})"
                )

            except Exception as exc:

                print(
                    f"  OCR FAILED on page {page}: {exc}"
                )

                # Keep the page in the result so that the
                # document ordering is preserved.
                page_text[page] = ""

    elapsed = perf_counter() - start

    print()
    print(
        f"OCR complete: {total_pages} page(s) "
        f"in {elapsed:.2f}s"
    )

    return page_text


# ==========================================================
# COMBINE ALL PAGES
# ==========================================================

def combine_all_pages(
    page_text: dict[int, str],
) -> str:
    """
    Combine OCR text from page 1 through the final page.

    Page boundaries are preserved with blank lines.
    """

    if not page_text:
        return ""

    parts = []

    for page_number in sorted(page_text):

        text = page_text.get(page_number, "").strip()

        if not text:
            continue

        parts.append(text)

    return "\n\n".join(parts).strip()


# ==========================================================
# PROCESS ONE DOCUMENT
# ==========================================================

def process_document(
    pdf_path: Path,
    cfg: OCRConfig | None = None,
) -> dict:
    """
    Process the ENTIRE PDF.

    Pipeline:

        PDF
         ↓
        Count pages
         ↓
        OCR page 1
        OCR page 2
        OCR page 3
        ...
        OCR last page
         ↓
        Combine all pages
         ↓
        Return complete OCR text
    """

    cfg = cfg or OCRConfig()

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    total_start = perf_counter()

    # ======================================================
    # OCR EVERY PAGE
    # ======================================================

    page_text = ocr_document(
        pdf_path,
        cfg,
    )

    # ======================================================
    # COMBINE
    # ======================================================

    raw_text = combine_all_pages(page_text)

    # ======================================================
    # CLEAN COMPLETE DOCUMENT
    # ======================================================

    cleaned_text = clean_ocr_text(raw_text)

    total_time = perf_counter() - total_start

    return {
        "pdf": str(pdf_path),
        "total_pages": len(page_text),
        "pages": page_text,
        "text": cleaned_text,
        "timing": {
            "total": round(total_time, 3),
        },
    }


# ==========================================================
# OCR PROCESSOR
# ==========================================================

class OCRProcessor:
    """
    Whole-document OCR processor.

    Usage:

        processor = OCRProcessor(
            tesseract_path=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        )

        text, output_path = processor.process(
            "document.pdf"
        )

    OR for Streamlit uploads:

        text = processor.process_bytes(
            pdf_bytes,
            "document.pdf"
        )

    Every page is OCR'd.
    """

    def __init__(
        self,
        poppler_path: str | None = None,
        output_folder: Path | str = "ocr_output",
        tesseract_path: str | None = None,
        dpi: int = 220,
        psm: int = 6,
        language: str = "eng",
        max_workers: int | None = None,
        denoise: bool = True,
    ) -> None:

        # Kept for compatibility with your existing code.
        # Poppler is NOT used.
        self.poppler_path = poppler_path

        self.output_folder = Path(
            output_folder
        )

        self.cfg = OCRConfig(
            dpi=dpi,
            psm=psm,
            language=language,
            denoise=denoise,
            **(
                {"max_workers": max_workers}
                if max_workers is not None
                else {}
            ),
        )

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = (
                tesseract_path
            )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ======================================================
    # SAVE TEXT
    # ======================================================

    def save_text(
        self,
        pdf_path: str | Path,
        text: str,
    ) -> Path:
        """
        Save complete OCR text as .txt.
        """

        stem = Path(pdf_path).stem or "document"

        output_path = (
            self.output_folder
            / f"{stem}.txt"
        )

        output_path.write_text(
            text,
            encoding="utf-8",
        )

        return output_path

    # ======================================================
    # PROCESS PDF PATH
    # ======================================================

    def process(
        self,
        pdf_path: str | Path,
        **overrides,
    ) -> tuple[str, Path]:
        """
        Process a complete PDF from disk.

        Returns:

            (
                complete_ocr_text,
                txt_output_path
            )
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        cfg = (
            replace(self.cfg, **overrides)
            if overrides
            else self.cfg
        )

        result = process_document(
            pdf_path,
            cfg,
        )

        text = result["text"]

        output_path = self.save_text(
            pdf_path,
            text,
        )

        return text, output_path

    # ======================================================
    # PROCESS UPLOADED BYTES
    # ======================================================

    def process_bytes(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf",
        **overrides,
    ) -> str:
        """
        Process a complete uploaded PDF.

        This is the method your Streamlit uploader can use.

        PDF bytes
            ↓
        temporary PDF
            ↓
        OCR EVERY PAGE
            ↓
        clean complete text
            ↓
        save .txt
            ↓
        return text
        """

        if not pdf_bytes:
            raise ValueError(
                "PDF bytes are empty."
            )

        cfg = (
            replace(self.cfg, **overrides)
            if overrides
            else self.cfg
        )

        # Only use the filename itself.
        # This prevents paths supplied through an uploaded
        # filename from escaping the output directory.
        safe_stem = (
            Path(
                Path(filename).name
            ).stem
            or "document"
        )

        suffix = (
            Path(filename).suffix
            or ".pdf"
        )

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_file.write(pdf_bytes)
            temp_path = Path(
                temp_file.name
            )

        try:

            result = process_document(
                temp_path,
                cfg,
            )

            # Complete OCR text.
            text = result["text"]

            # Save ONLY after the entire document
            # has been OCR'd and cleaned.
            output_path = (
                self.output_folder
                / f"{safe_stem}.txt"
            )

            output_path.write_text(
                text,
                encoding="utf-8",
            )

            return text

        finally:

            try:
                temp_path.unlink(
                    missing_ok=True
                )

            except OSError as exc:

                print(
                    f"  Could not remove temporary "
                    f"file {temp_path}: {exc}"
                )


# ==========================================================
# SIMPLE DIRECT USAGE
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # CHANGE THIS PATH
    # ------------------------------------------------------

    PDF_PATH = r"document.pdf"

    # ------------------------------------------------------
    # CHANGE THIS ONLY IF TESSERACT IS NOT IN PATH
    # ------------------------------------------------------

    TESSERACT_PATH = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    # ------------------------------------------------------
    # CREATE OCR PROCESSOR
    # ------------------------------------------------------

    processor = OCRProcessor(
        tesseract_path=TESSERACT_PATH,
        output_folder="ocr_output",
        dpi=220,
        psm=6,
        language="eng",
    )

    # ------------------------------------------------------
    # OCR THE WHOLE DOCUMENT
    # ------------------------------------------------------

    text, output_path = processor.process(
        PDF_PATH
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Characters: {len(text):,}")
    print()
    print(text[:5000])