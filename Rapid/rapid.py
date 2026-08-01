import os
import fitz
import numpy as np
from rapidocr import RapidOCR

PDF_PATH = "test.pdf"
START_PAGE = 2
END_PAGE = 12
DPI = 300

engine = RapidOCR()


def extract_text_from_image(image):
    result = engine(image)

    if result is None or result.txts is None:
        return ""

    return "\n".join(result.txts)


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    full_text = []

    start_index = START_PAGE - 1
    end_index = min(END_PAGE, len(document))

    for page_number in range(start_index, end_index):
        print(f"Processing page {page_number + 1}...")

        page = document[page_number]
        pix = page.get_pixmap(dpi=DPI)

        image = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(
            pix.height,
            pix.width,
            pix.n
        )

        if pix.n == 4:
            image = image[:, :, :3]

        text = extract_text_from_image(image)

        if text.strip():
            full_text.append(text)

    document.close()

    return "\n\n".join(full_text)


if __name__ == "__main__":
    text = extract_text_from_pdf(PDF_PATH)

    os.makedirs("OutputTXT", exist_ok=True)

    output_path = os.path.join("OutputTXT", "output.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)

    print("\nOCR completed.")
    print(f"Text saved to: {output_path}")