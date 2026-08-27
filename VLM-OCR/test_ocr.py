import ollama

IMAGE_PATH = "img.png"
MODEL = "qwen2.5vl:7b"

print("Running OCR...")

response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": """
Extract ALL visible text from this document exactly as written.

Rules:
- Do not summarize.
- Do not explain.
- Do not correct spelling.
- Do not infer missing text.
- Preserve numbers and punctuation.
- Preserve reading order.
- Return ONLY the extracted text.
""",
            "images": [IMAGE_PATH]
        }
    ]
)

print("\n" + "=" * 70)
print("OCR RESULT")
print("=" * 70)

print(response["message"]["content"])