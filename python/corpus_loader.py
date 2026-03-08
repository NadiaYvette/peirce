"""
Corpus Loader

Reads text files from the corpus/ directory structure, runs them through
the base model for embeddings, and saves chunked training data.

Directory structure:
  corpus/
    conversational/   — chat, dialogue, debate transcripts
    argumentative/    — philosophy, essays, logical reasoning
    narrative/        — prose fiction
    allegory/         — allegory, fable, parable
    poetry/           — verse
    drama/            — plays, scripts
    factual/          — encyclopedia, science, news

Each .txt file is read, split into paragraphs, embedded via Qwen,
and saved as .npy chunks with a manifest.

Usage:
    python3.12 python/corpus_loader.py [--chunk-size 64] [--corpus-dir corpus]
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from python.base_model import BaseModel


CORPUS_DIR = Path(__file__).parent.parent / "corpus"
CACHE_DIR = Path(__file__).parent.parent / "data" / "embeddings_cache"

CATEGORIES = [
    "conversational", "argumentative", "narrative",
    "allegory", "poetry", "drama", "factual"
]


def read_corpus(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    """
    Read all .txt files from corpus subdirectories.
    Returns list of {"text": ..., "category": ..., "source": ...}
    """
    documents = []
    for category in CATEGORIES:
        cat_dir = corpus_dir / category
        if not cat_dir.exists():
            continue
        for txt_file in sorted(cat_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            # Skip empty or very short files
            if len(text.strip()) < 50:
                continue
            documents.append({
                "text": text.strip(),
                "category": category,
                "source": txt_file.name,
            })
    return documents


def split_into_passages(text: str, max_chars: int = 2000, min_chars: int = 100) -> list[str]:
    """
    Split text into passages by paragraph boundaries.
    Tries to keep passages between min_chars and max_chars.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    passages = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and len(current) >= min_chars:
            passages.append(current)
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if len(current) >= min_chars:
        passages.append(current)
    elif current and passages:
        # Append short tail to last passage
        passages[-1] = passages[-1] + "\n\n" + current
    elif current:
        passages.append(current)

    return passages


def prepare_corpus(corpus_dir: Path = CORPUS_DIR, cache_dir: Path = CACHE_DIR,
                   chunk_size: int = 64, max_passages_per_doc: int = 50) -> list[dict]:
    """
    Full corpus preparation pipeline:
      1. Read all .txt files from corpus/
      2. Split into passages
      3. Embed each passage via Qwen
      4. Chunk into training examples
      5. Save to cache with manifest
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base model...")
    bm = BaseModel()
    print(f"Model: {bm.model.config._name_or_path}, d_model={bm.d_model}")

    documents = read_corpus(corpus_dir)
    print(f"\nFound {len(documents)} documents across categories:")
    cat_counts = {}
    for doc in documents:
        cat_counts[doc["category"]] = cat_counts.get(doc["category"], 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} files")

    manifest = []
    total_passages = 0
    total_tokens = 0

    for doc_idx, doc in enumerate(documents):
        passages = split_into_passages(doc["text"])
        if len(passages) > max_passages_per_doc:
            # Sample evenly across the document
            indices = np.linspace(0, len(passages) - 1, max_passages_per_doc, dtype=int)
            passages = [passages[i] for i in indices]

        for pass_idx, passage in enumerate(passages):
            try:
                embeddings, tokens = bm.get_embeddings(passage)
            except Exception as e:
                print(f"  Warning: skipping {doc['source']} passage {pass_idx}: {e}")
                continue

            d_model, T = embeddings.shape

            # Chunk long sequences
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                if end - start < 4:
                    continue

                chunk_embeddings = embeddings[:, start:end]
                chunk_tokens = tokens[start:end]

                chunk_id = f"doc{doc_idx:04d}_p{pass_idx:03d}_c{start:04d}"
                np.save(cache_dir / f"{chunk_id}_embeddings.npy", chunk_embeddings)

                entry = {
                    "id": chunk_id,
                    "doc_idx": doc_idx,
                    "passage_idx": pass_idx,
                    "start": start,
                    "num_tokens": end - start,
                    "tokens": chunk_tokens,
                    "embeddings_file": f"{chunk_id}_embeddings.npy",
                    "category": doc["category"],
                    "source": doc["source"],
                }
                manifest.append(entry)
                total_tokens += end - start

            total_passages += 1

        if (doc_idx + 1) % 5 == 0 or doc_idx == len(documents) - 1:
            print(f"  Processed {doc_idx + 1}/{len(documents)} documents "
                  f"({total_passages} passages, {total_tokens} tokens, {len(manifest)} chunks)")

    # Also include the original hardcoded training texts for continuity
    from python.data_loader import sample_training_texts
    hardcoded = sample_training_texts()
    print(f"\nAdding {len(hardcoded)} original training texts...")
    for i, text in enumerate(hardcoded):
        embeddings, tokens = bm.get_embeddings(text)
        d_model, T = embeddings.shape
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            if end - start < 4:
                continue
            chunk_id = f"orig{i:04d}_c{start:04d}"
            np.save(cache_dir / f"{chunk_id}_embeddings.npy", embeddings[:, start:end])
            manifest.append({
                "id": chunk_id,
                "doc_idx": len(documents) + i,
                "passage_idx": 0,
                "start": start,
                "num_tokens": end - start,
                "tokens": tokens[start:end],
                "embeddings_file": f"{chunk_id}_embeddings.npy",
                "category": "original",
                "source": f"hardcoded_{i}",
            })
            total_tokens += end - start

    # Save manifest
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Corpus Summary ===")
    print(f"  Documents: {len(documents)} + {len(hardcoded)} original")
    print(f"  Passages: {total_passages}")
    print(f"  Chunks: {len(manifest)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  d_model: {bm.d_model}")
    print(f"  Cached to: {cache_dir}")

    # Category breakdown in manifest
    cat_chunks = {}
    for entry in manifest:
        c = entry["category"]
        cat_chunks[c] = cat_chunks.get(c, 0) + 1
    print(f"\n  Chunks by category:")
    for cat, count in sorted(cat_chunks.items()):
        print(f"    {cat}: {count}")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare corpus for training")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--corpus-dir", type=str, default=str(CORPUS_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR))
    parser.add_argument("--max-passages", type=int, default=50,
                        help="Max passages per document")
    args = parser.parse_args()

    prepare_corpus(
        corpus_dir=Path(args.corpus_dir),
        cache_dir=Path(args.cache_dir),
        chunk_size=args.chunk_size,
        max_passages_per_doc=args.max_passages,
    )
