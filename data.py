import os
from pathlib import Path
import fitz  # PyMuPDF

# === Paths ===
raw_laws_dir = Path("legaldata/raw/laws_acts")
raw_cases_dir = Path("legaldata/raw/case_judgements")
output_file = Path("legaldata/final/Legal_corpus.txt")

# === Create output dir if needed ===
output_file.parent.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"[ERROR] Failed to read {pdf_path.name}: {e}")
        return ""

def convert_pdfs_to_txt():
    print(f"[INFO] Writing to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as outfile:
        for folder in [raw_laws_dir, raw_cases_dir]:
            if not folder.exists():
                print(f"[WARN] Folder not found: {folder}")
                continue
            for pdf_file in folder.rglob("*.pdf"):
                print(f"[PDF] Converting: {pdf_file.name}")
                text = extract_text_from_pdf(pdf_file)
                if text.strip():
                    outfile.write(f"\n\n===== {pdf_file.stem} =====\n\n")
                    outfile.write(text)
                else:
                    print(f"[SKIP] Empty or unreadable: {pdf_file.name}")
    print("[DONE] All PDFs combined into Legal_corpus.txt")

if __name__ == "__main__":
    convert_pdfs_to_txt()
