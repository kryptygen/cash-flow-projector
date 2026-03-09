import pandas as pd
import os
import io
import json
import base64
import re

from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from pdf2image import convert_from_bytes

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

REQUIRED_COLUMNS = {
    "operating": "Operating Cash Flow",
    "investing":  "Investing Cash Flow",
    "financing":  "Financing Cash Flow",
    "net":        "Net Cash Flow",
    "closing":    "Closing Balance",
    "year":       "Year",
    "month":      "Month",
}


# ── Column name normaliser ───────────────────────────────────────────────────
def smart_column_mapper(columns):
    mapped = {}
    for col in columns:
        c = col.lower().strip()
        if "operating" in c:
            mapped[col] = REQUIRED_COLUMNS["operating"]
        elif "invest" in c:
            mapped[col] = REQUIRED_COLUMNS["investing"]
        elif "financ" in c:
            mapped[col] = REQUIRED_COLUMNS["financing"]
        elif "net" in c:
            mapped[col] = REQUIRED_COLUMNS["net"]
        elif "closing" in c:
            mapped[col] = REQUIRED_COLUMNS["closing"]
        elif "year" in c:
            mapped[col] = REQUIRED_COLUMNS["year"]
        elif "month" in c:
            mapped[col] = REQUIRED_COLUMNS["month"]
    return mapped


# ── Single CSV / Excel file ──────────────────────────────────────────────────
def process_file(file):

    filename = file.filename.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    column_mapping = smart_column_mapper(df.columns)
    df = df.rename(columns=column_mapping)

    missing = [v for v in REQUIRED_COLUMNS.values() if v not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after mapping: {missing}")

    return df


# ── Dispatch by file type ────────────────────────────────────────────────────
def process_files(files):

    filenames = [f.filename.lower() for f in files]
    print("FILES RECEIVED:", [f.filename for f in files])

    # CSV
    if all(name.endswith(".csv") for name in filenames):
        dfs = [process_file(f) for f in files if f.filename]
        if not dfs:
            raise ValueError("No valid CSV files found.")
        return pd.concat(dfs, ignore_index=True)

    # Excel
    elif all(name.endswith((".xlsx", ".xls")) for name in filenames):
        dfs = [process_file(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    # Images
    elif all(name.endswith((".png", ".jpg", ".jpeg")) for name in filenames):
        images = []
        for file in files:
            file_bytes = file.read()
            image = Image.open(io.BytesIO(file_bytes))
            image.load()          # ensure fully decoded before seek
            file.seek(0)
            images.append(image)
        return extract_from_images(images)

    # PDF
    elif all(name.endswith(".pdf") for name in filenames):
        images = []
        for file in files:
            pdf_bytes = file.read()
            pages = convert_from_bytes(pdf_bytes)
            images.extend(pages)
            file.seek(0)
        return extract_from_images(images)

    else:
        raise ValueError(
            "All uploaded files must be the same format "
            "(CSV, Excel, images, or PDF)."
        )


# ── PIL Image → base64 part for Gemini ──────────────────────────────────────
def pil_to_part(image: Image.Image) -> dict:
    """Convert a PIL image to a Gemini inline_data part (base64 JPEG)."""
    buf = io.BytesIO()
    rgb = image.convert("RGB")
    rgb.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": b64,
        }
    }


# ── Gemini vision extraction ─────────────────────────────────────────────────
def extract_from_images(images: list) -> pd.DataFrame:

    prompt = """
Each image is a company's cash flow statement.

Extract ONLY these fields for every row / period visible:
  Year, Month,
  Operating Cash Flow, Investing Cash Flow, Financing Cash Flow,
  Net Cash Flow, Closing Balance

Return a valid JSON array and NOTHING else — no markdown fences, no explanation.
Example format:
[
  {
    "Year": 2023,
    "Month": "January",
    "Operating Cash Flow": 120000,
    "Investing Cash Flow": -30000,
    "Financing Cash Flow": -10000,
    "Net Cash Flow": 80000,
    "Closing Balance": 500000
  }
]

Rules:
- Use null for any value you cannot read clearly.
- Do NOT guess or invent values.
- Return ONLY the JSON array.
"""

    # Build content list: image parts first, then the text prompt
    contents = [{"text": prompt}]
    contents += [pil_to_part(img) for img in images]

    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(contents)

    raw = response.text.strip()
    print("Gemini raw response : ",raw)
    # Strip markdown fences if Gemini adds them anyway
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini returned invalid JSON. Parse error: {e}\n"
            f"Raw response (first 500 chars): {raw[:500]}"
        )

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            "Gemini returned an empty or non-list response. "
            "Check that the images contain readable cash flow data."
        )

    df = pd.DataFrame(data)

    # Ensure required columns exist (fill missing ones with None)
    for col in REQUIRED_COLUMNS.values():
        if col not in df.columns:
            df[col] = None

    return df