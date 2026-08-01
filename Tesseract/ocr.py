import cv2
import fitz
import pytesseract
import numpy as np
import os

PDF_PATH = "test.pdf"
START_PAGE = 2
END_PAGE = 12
DPI = 300
MIN_CONFIDENCE = 82


def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray, lang="eng", output_type=pytesseract.Output.DICT
    )

    lines = {}

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        confidence = float(data["conf"][i])

        if text and confidence >= MIN_CONFIDENCE:
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])

            if key not in lines:
                lines[key] = []

            lines[key].append(text)

    output = []

    for words in lines.values():
        output.append(" ".join(words))

    return "\n".join(output)


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    full_text = []

    start_index = START_PAGE - 1
    end_index = min(END_PAGE, len(document))

    for page_number in range(start_index, end_index):
        print(f"Processing page {page_number + 1}...")

        page = document[page_number]
        pix = page.get_pixmap(dpi=DPI)

        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        if pix.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        text = extract_text_from_image(image)

        if text:
            full_text.append(text)

    document.close()

    return "\n\n".join(full_text)


if __name__ == "__main__":
    text = extract_text_from_pdf(PDF_PATH)

    output_folder = "OutputTXT"
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "output.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"\nOCR completed.")
    print(f"Text saved to: {output_path}")
