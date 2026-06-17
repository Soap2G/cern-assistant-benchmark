#!/usr/bin/env python3
"""CDS live search utility — helps populate gold_recids in task YAML files.

Uses cds.cern.ch old-Invenio search with MARC XML output (of=xm).

Usage:
  python tools/cds_query.py "ATLAS Higgs boson 2012"
  python tools/cds_query.py --collection "ATLAS Papers" "top quark mass 2023" --size 10
  python tools/cds_query.py --author "ATLAS Collaboration" --year 2024 "single top"
  python tools/cds_query.py --ids-only "HTCondor batch" --size 5

Outputs recids + titles so you can verify and paste into task YAML.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

CDS_SEARCH_BASE = "https://cds.cern.ch/search"
MARC_NS = "http://www.loc.gov/MARC21/slim"


def _build_query(query: str, collection: str | None, author: str | None, year: int | None) -> str:
    parts = [query] if query else []
    if author:
        parts.append(f'author:"{author}"')
    if year:
        parts.append(f"year:{year}")
    return " AND ".join(parts) if parts else "*"


def _fetch_xml(query: str, collection: str | None, size: int) -> str | None:
    """Fetch MARC XML from CDS old-Invenio search endpoint."""
    params: dict[str, Any] = {
        "p": query,
        "of": "xm",
        "rg": size,
        "action_search": "Search",
        "sf": "latest first",
        "rm": "",
        "sc": 0,
    }
    if collection:
        params["c"] = collection

    url = f"{CDS_SEARCH_BASE}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/xml"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        print(f"HTTP error {e.code}: {e.reason}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return None


def _fetch_ids(query: str, collection: str | None, size: int) -> list[int]:
    """Fetch just recids using of=id (returns bare list)."""
    params: dict[str, Any] = {
        "p": query,
        "of": "id",
        "rg": size,
        "action_search": "Search",
    }
    if collection:
        params["c"] = collection

    url = f"{CDS_SEARCH_BASE}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace").strip()
        # Old Invenio returns Python list literal: [123, 456, ...]
        import ast
        return ast.literal_eval(body)
    except Exception as e:
        print(f"ID fetch failed: {e}", file=sys.stderr)
        return []


def _parse_marc_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse MARC XML response into list of record dicts."""
    # Strip XML declaration before parsing
    if xml_text.lstrip().startswith("<?xml"):
        xml_text = xml_text[xml_text.index("?>") + 2:]

    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError as e:
        print(f"XML parse error: {e}", file=sys.stderr)
        return []

    ns = {"m": MARC_NS}
    results = []

    for rec in root.findall(".//m:record", ns):
        recid = None
        title = ""
        date = ""
        report = ""

        for cf in rec.findall("m:controlfield", ns):
            if cf.get("tag") == "001":
                try:
                    recid = int(cf.text or "")
                except ValueError:
                    pass

        for df in rec.findall("m:datafield", ns):
            tag = df.get("tag", "")
            subs = {sf.get("code"): (sf.text or "") for sf in df.findall("m:subfield", ns)}

            if tag == "245" and not title:
                title = subs.get("a", "")
            if tag == "269" and not date:
                date = subs.get("c", "")
            if tag == "260" and not date:
                date = subs.get("c", "")
            if tag == "037" and not report:
                report = subs.get("a", "")

        if recid is not None:
            results.append({
                "recid": recid,
                "title": title,
                "date": date[:10],
                "report_number": report,
            })

    return results


def search_cds(
    query: str,
    collection: str | None = None,
    author: str | None = None,
    year: int | None = None,
    size: int = 10,
) -> list[dict[str, Any]]:
    """Query CDS and return list of dicts with recid, title, date, report_number."""
    q = _build_query(query, collection, author, year)
    xml_text = _fetch_xml(q, collection, size)
    if not xml_text:
        return []
    return _parse_marc_xml(xml_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query live CDS for gold recids")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument("--collection", "-c", help="Restrict to a collection (e.g. 'ATLAS Papers')")
    parser.add_argument("--author", help="Filter by author/collaboration")
    parser.add_argument("--year", type=int, help="Filter by year")
    parser.add_argument("--size", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--ids-only", action="store_true",
                        help="Output only a bare list of recids")
    parser.add_argument("--yaml-snippet", action="store_true",
                        help="Output as gold_recids YAML snippet")
    args = parser.parse_args()

    q = _build_query(args.query, args.collection, args.author, args.year)
    print(f"Searching CDS: {q!r}", file=sys.stderr)

    if args.ids_only:
        ids = _fetch_ids(q, args.collection, args.size)
        if not ids:
            print("No results found.", file=sys.stderr)
            sys.exit(1)
        print(ids)
        return

    results = search_cds(
        query=args.query,
        collection=args.collection,
        author=args.author,
        year=args.year,
        size=args.size,
    )

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    if args.yaml_snippet:
        print("gold_recids:")
        for r in results:
            print(f"  - {r['recid']}  # {r['title'][:60]}")
        return

    print(f"{'recid':>10}  {'date':>10}  {'report':>22}  title")
    print("-" * 90)
    for r in results:
        recid = str(r.get("recid", "?"))
        title = str(r.get("title", ""))[:45]
        date = str(r.get("date", ""))
        report = str(r.get("report_number", ""))[:22]
        print(f"{recid:>10}  {date:>10}  {report:>22}  {title}")


if __name__ == "__main__":
    main()
