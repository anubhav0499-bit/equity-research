"""
Equity Research Platform — CLI Entry Point
==========================================
Usage:
    python main.py "Apple"
    python main.py "HDFC Bank" --ticker HDFCBANK
    python main.py "Infosys" "TCS" --output /path/to/reports
    python main.py --batch companies.txt
    python main.py --check
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║         EQUITY INTELLIGENCE RESEARCH PLATFORM v1.0              ║
║   Autonomous Institutional-Grade Equity Research System          ║
║   17-Agent Architecture | DCF | Forensics | DOCX Reports         ║
╚══════════════════════════════════════════════════════════════════╝
"""


def _print_result(result: dict) -> None:
    print(f"\n{'─'*70}")
    print(f"  Company:         {result.get('company_name')}")
    print(f"  Ticker:          {result.get('ticker')}")
    print(f"  Rating:          {result.get('investment_rating', 'N/A')}")
    print(f"  Target Price:    {result.get('target_price', 'N/A')}")
    print(f"  Current Price:   {result.get('current_price', 'N/A')}")
    upside = result.get("upside_pct")
    if upside is not None:
        print(f"  Upside/Downside: {upside:+.1f}%")
    print(f"  Risk Score:      {result.get('overall_risk_score', 0):.0f}/100 ({result.get('overall_risk_classification', 'N/A')})")
    print(f"  Critical Flags:  {result.get('critical_findings', 0)}")
    print(f"  Agents Run:      {result.get('agents_completed', 0)}")
    print(f"  Duration:        {result.get('duration_seconds', 0):.0f}s")
    print(f"  Validation:      {'PASSED' if result.get('validation_passed') else 'FAILED'}")
    if result.get("report_path"):
        print(f"  Report:          {result['report_path']}")
    if result.get("error"):
        print(f"  ERROR:           {result['error']}")
    print(f"{'─'*70}\n")


def _check_requirements() -> None:
    packages = [
        ("yfinance",            "yfinance"),
        ("pandas",              "pandas"),
        ("numpy",               "numpy"),
        ("pydantic",            "pydantic"),
        ("docx",                "python-docx"),
        ("httpx",               "httpx"),
        ("loguru",              "loguru"),
        ("bs4",                 "beautifulsoup4"),
        ("dotenv",              "python-dotenv"),
        ("openai",              "openai"),
        ("anthropic",           "anthropic"),
        ("groq",                "groq"),
    ]
    print("\n  Package Status:")
    all_ok = True
    for import_name, pip_name in packages:
        try:
            __import__(import_name)
            print(f"    ✓  {pip_name}")
        except ImportError:
            print(f"    ✗  {pip_name}  →  pip install {pip_name}")
            all_ok = False
    print()
    if not all_ok:
        print("  Install all: pip install -r requirements.txt\n")
    else:
        print("  All core dependencies installed.\n")


def main() -> None:
    print(BANNER)
    parser = argparse.ArgumentParser(
        description="Equity Intelligence Research Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Apple"
  python main.py "HDFC Bank" --ticker HDFCBANK
  python main.py "Infosys" "TCS" --batch extra.txt
  python main.py --check
        """,
    )
    parser.add_argument("companies", nargs="*", help="Company names to research")
    parser.add_argument("--ticker", "-t", help="Ticker symbol hint for single company")
    parser.add_argument("--batch", "-b", metavar="FILE", help="File with one company per line")
    parser.add_argument("--output", "-o", metavar="DIR", help="Output directory override")
    parser.add_argument("--check", action="store_true", help="Check dependencies and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.check:
        _check_requirements()
        sys.exit(0)

    companies = list(args.companies or [])
    if args.batch:
        path = Path(args.batch)
        if path.exists():
            companies.extend(l.strip() for l in path.read_text().splitlines()
                             if l.strip() and not l.startswith("#"))

    if not companies:
        parser.print_help()
        print("\n  No companies specified.\n")
        sys.exit(1)

    # Deduplicate
    seen: set = set()
    companies = [c for c in companies if not (c in seen or seen.add(c))]

    # Setup logging
    from equity_research.core.logging_setup import setup_logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Load orchestrator
    try:
        from equity_research.orchestrator.workflow import ResearchOrchestrator
        orchestrator = ResearchOrchestrator()
    except Exception as e:
        print(f"\n  Failed to initialise orchestrator: {e}")
        print("  Ensure you have installed dependencies: pip install -r requirements.txt")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else None
    all_results = []

    print(f"  Researching {len(companies)} {'company' if len(companies)==1 else 'companies'}:\n")
    for i, company in enumerate(companies, 1):
        print(f"  [{i}/{len(companies)}] {company} — {datetime.now().strftime('%H:%M:%S')}")
        try:
            ticker = args.ticker if (len(companies) == 1 and args.ticker) else ""
            result = orchestrator.research(company, ticker=ticker, output_dir=output_dir)
            all_results.append(result)
            _print_result(result)
        except KeyboardInterrupt:
            print("\n  Research interrupted.")
            break
        except Exception as e:
            print(f"  Error researching {company}: {e}")
            all_results.append({"company_name": company, "error": str(e)})

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'═'*70}")
        print(f"  BATCH SUMMARY — {len(all_results)} companies")
        print(f"{'─'*70}")
        print(f"  {'Company':<35} {'Rating':<12} {'Target':>8}  {'Risk':>5}")
        print(f"  {'─'*35} {'─'*12} {'─'*8}  {'─'*5}")
        for r in all_results:
            name = r.get("company_name", "?")[:33]
            rating = r.get("investment_rating", "ERROR")[:10]
            target = f"{r.get('target_price', 0):.2f}" if r.get("target_price") else "N/A"
            risk = f"{r.get('overall_risk_score', 0):.0f}"
            print(f"  {name:<35} {rating:<12} {target:>8}  {risk:>5}")
        print(f"{'═'*70}\n")

    failed = sum(1 for r in all_results if r.get("error"))
    sys.exit(failed)


if __name__ == "__main__":
    main()
