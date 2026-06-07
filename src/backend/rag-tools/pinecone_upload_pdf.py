"""CLI: chunk a reference PDF and upsert section-aware embeddings into Pinecone.

Usage:
    python rag-tools/pinecone_upload_pdf.py <pdf_path> <namespace> [--index NAME]
        [--source SLUG] [--book-title "Title"] [--page-offset N]
        [--chunk-tokens 800] [--overlap-tokens 100] [--embed-batch 100]
        [--replace] [--dry-run]

--page-offset converts PDF page position to printed page (e.g. PDF p.15 shows
printed p.1 -> --page-offset 14). Env: PINECONE_API_KEY, OPENAI_API_KEY,
PINECONE_INDEX_NAME (if --index omitted). The index MUST be 1536-dimensional.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the importable `knowledge` package reachable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from knowledge.chunking import extract_pages, section_aware_chunks  # noqa: E402
from knowledge.ingest import delete_existing_source, ingest_chunks, preflight, slugify  # noqa: E402


def _print_section_diagnostic(chunks) -> None:
    if not chunks:
        print("      WARNING: no chunks produced — check extraction/chunking.")
        return
    non_empty = [c for c in chunks if c.section_path.strip()]
    pct = 100.0 * len(non_empty) / len(chunks)
    print(f"      section_path non-empty: {len(non_empty)}/{len(chunks)} ({pct:.0f}%)")
    print("      sample section paths:")
    for c in chunks[:5]:
        print(f"        p.{c.page_start}-{c.page_end}: {c.section_path or '(none)'}")
    if pct < 20:
        print(
            "      WARNING: very few chunks have a section path — the PDF may not emit "
            "markdown headers. 'section-aware' is degrading to token windows."
        )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument(
        "namespace",
        nargs="?",
        default=os.getenv("REFERENCE_NAMESPACE"),
        help="Pinecone namespace (default: $REFERENCE_NAMESPACE). Must match what the app queries.",
    )
    parser.add_argument("--index", default=os.getenv("PINECONE_INDEX_NAME"))
    parser.add_argument("--source", default=None, help="ID-safe source slug (default: slugified PDF stem)")
    parser.add_argument("--book-title", default=None, help="Human-readable title (default: PDF stem)")
    parser.add_argument("--page-offset", type=int, default=0, help="PDF page minus printed page (front-matter offset)")
    parser.add_argument("--chunk-tokens", type=int, default=800)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    parser.add_argument("--embed-batch", type=int, default=100)
    parser.add_argument("--replace", action="store_true", help="Delete existing vectors for this source first")
    parser.add_argument("--dry-run", action="store_true", help="Extract + chunk only; skip embed/upsert")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"ERROR: PDF not found: {args.pdf_path}", file=sys.stderr)
        return 2
    if not args.index:
        print("ERROR: --index not given and PINECONE_INDEX_NAME not set", file=sys.stderr)
        return 2
    if not args.namespace:
        print("ERROR: no namespace given and REFERENCE_NAMESPACE not set", file=sys.stderr)
        return 2

    source = slugify(args.source) if args.source else slugify(args.pdf_path.stem)
    book_title = args.book_title or args.pdf_path.stem

    print(f"[extract] Extracting pages: {args.pdf_path}")
    pages = extract_pages(str(args.pdf_path))
    pages_text = "".join(p.markdown for p in pages)
    print(f"      {len(pages)} pages, {len(pages_text):,} chars")

    print("[chunk] Chunking (section-aware)")
    chunks = section_aware_chunks(
        pages,
        source=source,
        book_title=book_title,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        page_offset=args.page_offset,
    )
    print(f"      {len(chunks):,} chunks (source slug='{source}')")
    _print_section_diagnostic(chunks)

    if args.dry_run:
        print(f"DRY RUN — would upsert {len(chunks):,} vectors to ns='{args.namespace}'")
        return 0

    if not os.getenv("PINECONE_API_KEY"):
        print("ERROR: PINECONE_API_KEY not set", file=sys.stderr)
        return 2

    from pinecone import Pinecone

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    try:
        preflight(pc, args.index, pages_text=pages_text, page_count=len(pages))
    except Exception as e:  # noqa: BLE001 — surface a friendly precondition error
        print(f"ERROR: pre-flight failed: {e}", file=sys.stderr)
        print("       Confirm the 1536-d index exists and PINECONE_INDEX_NAME is correct.", file=sys.stderr)
        return 2
    index = pc.Index(args.index)

    if args.replace:
        print(f"[replace] Replacing existing vectors for source='{source}'")
        delete_existing_source(index, source=source, namespace=args.namespace)

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    print(f"[upload] Embedding + upserting to ns='{args.namespace}'")
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    count = ingest_chunks(
        chunks,
        source=source,
        book_title=book_title,
        namespace=args.namespace,
        index=index,
        openai_client=openai_client,
        batch_size=args.embed_batch,
    )
    print(f"Done. ns='{args.namespace}' now has {count:,} vectors from '{source}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
