"""
PDF Text Extraction

Extracts text from PDF files for corpus building.
Handles both text-based and OCR'd PDFs (via PyMuPDF/fitz).

Usage:
    # Single file
    python3.12 python/pdf_extract.py input.pdf output.txt

    # Directory of PDFs → text files
    python3.12 python/pdf_extract.py --dir pdfs/ --out corpus/factual/

    # Download PDF from URL, extract, save
    python3.12 python/pdf_extract.py --url https://example.com/book.pdf --out corpus/narrative/book.txt
"""

import sys
import argparse
from pathlib import Path

try:
    import pymupdf
except ImportError:
    print("pymupdf not installed. Run: pip3.12 install pymupdf", file=sys.stderr)
    sys.exit(1)


def extract_text_from_pdf(pdf_path: str | Path, max_pages: int = 0) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: path to the PDF file
        max_pages: maximum pages to extract (0 = all)

    Returns:
        Extracted text as a single string with page breaks as double newlines.
    """
    doc = pymupdf.open(str(pdf_path))
    pages = []
    n = len(doc) if max_pages == 0 else min(max_pages, len(doc))

    for i in range(n):
        page = doc[i]
        text = page.get_text()
        if text.strip():
            pages.append(text.strip())

    doc.close()
    return "\n\n".join(pages)


def clean_extracted_text(text: str) -> str:
    """
    Clean up common PDF extraction artifacts.
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        # Skip page numbers (lines that are just a number)
        stripped = line.strip()
        if stripped.isdigit() and len(stripped) <= 4:
            continue
        # Skip very short lines that are likely headers/footers
        # (but keep them if they could be poetry lines)
        cleaned.append(line)

    text = "\n".join(cleaned)

    # Collapse excessive whitespace
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


def download_and_extract(url: str, output_path: str | Path, max_pages: int = 0) -> str:
    """
    Download a PDF from a URL and extract its text.
    """
    import urllib.request
    import tempfile

    print(f"Downloading: {url}", file=sys.stderr)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        tmp_path = tmp.name

    text = extract_text_from_pdf(tmp_path, max_pages)
    text = clean_extracted_text(text)

    Path(tmp_path).unlink()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")
        print(f"Saved: {output_path} ({len(text)} chars, {text.count(chr(10))} lines)",
              file=sys.stderr)

    return text


def process_directory(input_dir: str | Path, output_dir: str | Path,
                      max_pages: int = 0) -> list[Path]:
    """
    Extract text from all PDFs in a directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for pdf_file in sorted(input_dir.glob("*.pdf")):
        out_file = output_dir / (pdf_file.stem + ".txt")
        print(f"Processing: {pdf_file.name}", file=sys.stderr)
        try:
            text = extract_text_from_pdf(pdf_file, max_pages)
            text = clean_extracted_text(text)
            out_file.write_text(text, encoding="utf-8")
            print(f"  → {out_file.name} ({len(text)} chars)", file=sys.stderr)
            outputs.append(out_file)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs")
    parser.add_argument("input", nargs="?", help="Input PDF file")
    parser.add_argument("output", nargs="?", help="Output text file")
    parser.add_argument("--url", help="Download PDF from URL")
    parser.add_argument("--dir", help="Process all PDFs in directory")
    parser.add_argument("--out", help="Output directory (with --dir) or file (with --url)")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="Maximum pages to extract (0 = all)")
    args = parser.parse_args()

    if args.url:
        output = args.out or args.output or "extracted.txt"
        text = download_and_extract(args.url, output, args.max_pages)
        print(f"Extracted {len(text)} characters")

    elif args.dir:
        output_dir = args.out or "extracted/"
        outputs = process_directory(args.dir, output_dir, args.max_pages)
        print(f"Processed {len(outputs)} PDF files")

    elif args.input:
        text = extract_text_from_pdf(args.input, args.max_pages)
        text = clean_extracted_text(text)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Saved: {args.output} ({len(text)} chars)", file=sys.stderr)
        else:
            print(text)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
