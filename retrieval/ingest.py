"""
RAG Ingestion CLI — manually add documents to a company's vector store.

Documents ingested here are indexed alongside those automatically collected
by FilingRetrievalAgent and TranscriptRetrievalAgent during a research run.

Usage
-----
# Index a PDF filing
python -m equity_research.retrieval.ingest --ticker AAPL --file ./apple_10k_2024.pdf

# Index a web page (IR page, news article, analyst report)
python -m equity_research.retrieval.ingest --ticker AAPL --url https://ir.apple.com/...

# Index raw text (paste a transcript excerpt, press release, etc.)
python -m equity_research.retrieval.ingest --ticker AAPL --text "Revenue for Q3 FY2024..."

# Index an entire folder of filings
python -m equity_research.retrieval.ingest --ticker AAPL --dir ./filings/apple/

# Check how many documents are indexed for a ticker
python -m equity_research.retrieval.ingest --ticker AAPL --status

# Clear all indexed documents for a ticker
python -m equity_research.retrieval.ingest --ticker AAPL --clear
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from loguru import logger

# Ensure the package root is importable when run as __main__
_pkg_root = Path(__file__).parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from equity_research.retrieval.vector_store import (
    ingest_document,
    ingest_texts,
    query,
    collection_size,
    clear_company,
)


def _extract_text_from_file(path: Path) -> str:
    """Extract plain text from PDF, TXT, MD, or HTML files."""
    suffix = path.suffix.lower()
    raw = path.read_bytes()

    if suffix == ".pdf":
        # pdfminer.six (preferred) → pypdf → raw bytes decode
        try:
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams
            import io
            out = io.StringIO()
            extract_text_to_fp(io.BytesIO(raw), out, laparams=LAParams())
            text = out.getvalue().strip()
            if text:
                return text
        except Exception:
            pass
        try:
            import pypdf, io
            reader = pypdf.PdfReader(io.BytesIO(raw))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            pass
        return raw.decode("utf-8", errors="ignore")

    if suffix in (".html", ".htm"):
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(raw, "html.parser").get_text(separator="\n")
        except Exception:
            return raw.decode("utf-8", errors="ignore")

    # .txt, .md, and anything else
    return raw.decode("utf-8", errors="ignore")


def _fetch_url(url: str) -> str:
    """Fetch and extract text from a URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
        r = requests.get(url, timeout=30, headers={"User-Agent": "EquityResearchPlatform/1.0"})
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "pdf" in ct:
            # Inline PDF — save to temp bytes and extract
            import io
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams
            out = io.StringIO()
            extract_text_to_fp(io.BytesIO(r.content), out, laparams=LAParams())
            return out.getvalue().strip()
        return BeautifulSoup(r.text, "html.parser").get_text(separator="\n")
    except Exception as e:
        logger.error(f"URL fetch failed for {url}: {e}")
        return ""


def _ingest_file(path: Path, ticker: str, meta: dict) -> int:
    text = _extract_text_from_file(path)
    if not text.strip():
        print(f"  WARNING: no text extracted from {path.name}")
        return 0
    meta = {"filename": path.name, "suffix": path.suffix, **meta}
    n = ingest_document(text[:80000], meta, ticker)
    print(f"  {path.name}: {n} chunks indexed")
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Equity Research RAG — ingest documents into a company's vector store"
    )
    parser.add_argument("--ticker", required=True,
                        help="Company ticker (e.g. AAPL, RELIANCE, HDFCBANK)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file",   help="Path to a PDF, TXT, MD, or HTML file")
    group.add_argument("--url",    help="URL of a web page or PDF to fetch and index")
    group.add_argument("--text",   help="Raw text string to index directly")
    group.add_argument("--dir",    help="Directory of files to ingest recursively")
    group.add_argument("--status", action="store_true",
                       help="Show number of indexed documents for this ticker")
    group.add_argument("--clear",  action="store_true",
                       help="Remove all indexed documents for this ticker")

    parser.add_argument("--metadata", default="{}",
                        help='Optional JSON metadata, e.g. \'{"source": "10-K", "year": "2024"}\'')
    parser.add_argument("--query",  dest="test_query",
                        help="Test retrieval with a question after ingestion")

    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    try:
        meta = json.loads(args.metadata)
    except json.JSONDecodeError:
        print(f"ERROR: --metadata is not valid JSON: {args.metadata}")
        sys.exit(1)

    # ── Status ────────────────────────────────────────────────────────────
    if args.status:
        n = collection_size(ticker)
        print(f"{ticker}: {n} chunks indexed in vector store")
        return

    # ── Clear ─────────────────────────────────────────────────────────────
    if args.clear:
        confirm = input(f"Delete all indexed documents for {ticker}? [y/N] ").strip().lower()
        if confirm == "y":
            clear_company(ticker)
            print(f"{ticker}: vector store cleared")
        else:
            print("Cancelled.")
        return

    # ── Ingest ────────────────────────────────────────────────────────────
    total = 0

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"ERROR: file not found: {args.file}")
            sys.exit(1)
        total = _ingest_file(path, ticker, meta)

    elif args.url:
        print(f"Fetching {args.url} ...")
        text = _fetch_url(args.url)
        if not text.strip():
            print("ERROR: no text extracted from URL")
            sys.exit(1)
        n = ingest_document(text[:80000], {"url": args.url, **meta}, ticker)
        print(f"  URL: {n} chunks indexed")
        total = n

    elif args.text:
        n = ingest_document(args.text, {"source": "raw_text", **meta}, ticker)
        print(f"  Raw text: {n} chunks indexed")
        total = n

    elif args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"ERROR: directory not found: {args.dir}")
            sys.exit(1)
        supported = {".pdf", ".txt", ".md", ".html", ".htm"}
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in supported)
        if not files:
            print(f"No supported files found in {args.dir}")
            return
        print(f"Ingesting {len(files)} files from {d} ...")
        for p in files:
            try:
                total += _ingest_file(p, ticker, meta)
            except Exception as e:
                print(f"  {p.name}: FAILED ({e})")

    else:
        parser.print_help()
        return

    print(f"\nTotal chunks indexed: {total}")
    print(f"Vector store size for {ticker}: {collection_size(ticker)} chunks")

    # ── Optional test query ───────────────────────────────────────────────
    if args.test_query and total > 0:
        print(f"\nTest query: {args.test_query!r}")
        from equity_research.retrieval.vector_store import query as vs_query
        results = vs_query(args.test_query, ticker=ticker, top_k=3)
        for i, chunk in enumerate(results, 1):
            print(f"\n[{i}] {chunk[:300]}{'...' if len(chunk) > 300 else ''}")


if __name__ == "__main__":
    main()
