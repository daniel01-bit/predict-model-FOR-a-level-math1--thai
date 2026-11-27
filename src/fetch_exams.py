"""Discover and download publicly accessible Thai Math exam PDFs (Tier 1 & Tier 3)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin

import requests


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

SEED_PAGES = [
    "https://www.rathcenter.com/",
    "https://www.rathcenter.com/oldweb/Exam/Entrance.html",
    "https://www.rathcenter.com/oldweb/Exam/Saman.html",
    "https://www.rathcenter.com/oldweb/Exam/ANET.html",
    "https://www.1clickmath.com/",
    "https://www.meddentgat.com/category/math-alevel/",
    "https://www.jacksupermath.in.th/",
]


def discover_links(urls: list[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for url in urls:
        try:
            resp = requests.get(url, timeout=12, headers=HEADERS)
            resp.encoding = resp.apparent_encoding or "utf-8"
            html_text = resp.text
            hrefs = re.findall(r'href=["\']([^"\']+)["\']', html_text, re.I)
            raw_drives = re.findall(r'https?://drive\.google\.com/[^\s"\'<>\\]+', html_text, re.I)
            raw_pdfs = re.findall(r'https?://[^\s"\'<>\\]+\.pdf(?:\?[^\s"\'<>\\]*)?', html_text, re.I)
            pdfs = sorted({urljoin(url, h) for h in hrefs if ".pdf" in h.lower()} | set(raw_pdfs))
            drives = sorted({h for h in hrefs if "drive.google.com" in h.lower()} | set(raw_drives))
            subpages = sorted(
                {
                    urljoin(url, h)
                    for h in hrefs
                    if any(k in h.lower() for k in ("a-level", "alevel", "saman", "exam", "ent", "math1", "math-1", "64", "65", "66", "67", "68", "69"))
                    and not h.startswith("#")
                    and not h.startswith("javascript:")
                }
            )
            found[url] = {
                "status": resp.status_code,
                "pdfs": pdfs,
                "drives": drives,
                "subpages": subpages[:15],
            }
        except Exception as exc:
            found[url] = {"error": str(exc)}
    return found


def download_pdf(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Convert Google Drive share URL to direct download URL if needed
    m = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        url = f"https://drive.google.com/uc?export=download&id={m.group(1)}"

    resp = requests.get(url, timeout=30, headers=HEADERS, stream=True)
    if not resp.ok:
        return False
    data = resp.content
    if not data.startswith(b"%PDF-"):
        return False
    dest.write_bytes(data)
    return True


def extract_drive_entries_with_labels(url: str) -> list[tuple[str, str]]:
    """Extract unique Google Drive file IDs with nearby Thai/English labels from a page."""
    resp = requests.get(url, timeout=15, headers=HEADERS)
    resp.encoding = resp.apparent_encoding or "utf-8"
    text = resp.text

    seen: set[str] = set()
    entries: list[tuple[str, str]] = []
    for m in re.finditer(r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", text):
        fid = m.group(1)
        if fid in seen:
            continue
        seen.add(fid)
        start = max(0, m.start() - 260)
        end = min(len(text), m.end() + 180)
        snippet = re.sub(r"<[^>]+>", " ", text[start:end])
        snippet = re.sub(r"\s+", " ", snippet).strip()
        # Try to extract clean Thai/year label
        label_match = re.findall(r"[\u0E00-\u0E7FA-Za-z0-9\-\(\)\s]{4,80}", snippet)
        clean_label = " | ".join(s.strip() for s in label_match if any(c in s for c in ("6", "5", "4", "ปี", "คณิต", "A-Level", "สามัญ", "Ent", "ข้อสอบ", "เฉลย")))[:100]
        entries.append((fid, clean_label or snippet[:80]))
    return entries


def _extract_filename_from_headers(resp: requests.Response, fallback: str) -> str:
    from urllib.parse import unquote

    cd = resp.headers.get("Content-Disposition", "")
    m_star = re.search(r"filename\*=UTF-8''([^;]+)", cd, re.I)
    if m_star:
        return unquote(m_star.group(1)).strip().strip('"')
    m_plain = re.search(r'filename="?([^";]+)"?', cd, re.I)
    if m_plain:
        return m_plain.group(1).strip()
    return fallback


def _detect_year_be(name: str) -> int | None:
    m4 = re.search(r"(25[3-6][0-9])", name)
    if m4:
        return int(m4.group(1))
    m2 = re.search(r"(?:^|[^0-9])(6[4-9]|5[5-9]|6[0-3]|4[1-9]|5[0-4])(?:[^0-9]|$)", name)
    if m2:
        return 2500 + int(m2.group(1))
    return None


def download_drive_page_pdfs(page_url: str, raw_root: Path) -> list[dict]:
    entries = extract_drive_entries_with_labels(page_url)
    downloaded: list[dict] = []
    for idx, (fid, _) in enumerate(entries, start=1):
        dl_url = f"https://drive.google.com/uc?export=download&id={fid}"
        try:
            resp = requests.get(dl_url, timeout=40, headers=HEADERS)
            if not resp.ok or not resp.content.startswith(b"%PDF-"):
                print(f"[{idx:02d}/{len(entries)}] SKIP {fid} (not direct PDF, status={resp.status_code})", flush=True)
                continue
            orig_name = _extract_filename_from_headers(resp, f"{fid}.pdf")
            safe_name = re.sub(r'[<>:"/\\|?*]', "_", orig_name)
            if not safe_name.lower().endswith(".pdf"):
                safe_name += ".pdf"
            yr = _detect_year_be(safe_name)
            if yr is not None and 2564 <= yr <= 2569:
                subdir = raw_root / "tier1_alevel_64_69"
                tier = "tier_1_target"
            else:
                subdir = raw_root / "tier3_saman_ent"
                tier = "tier_3_legacy"
            subdir.mkdir(parents=True, exist_ok=True)
            dest = subdir / safe_name
            dest.write_bytes(resp.content)
            mb = len(resp.content) / (1024 * 1024)
            print(f"[{idx:02d}/{len(entries)}] OK {tier} yr={yr} ({mb:.2f} MB) -> {dest.name}", flush=True)
            downloaded.append({"file": str(dest), "tier": tier, "year_be": yr, "size_mb": round(mb, 2)})
        except Exception as exc:
            print(f"[{idx:02d}/{len(entries)}] ERR {fid}: {exc}", flush=True)
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--list-drive", type=str, default=None)
    parser.add_argument("--download-page", type=str, default=None)
    parser.add_argument("--url", action="append", default=[])
    args = parser.parse_args()

    if args.download_page:
        raw_root = Path(__file__).resolve().parent.parent / "data" / "raw"
        items = download_drive_page_pdfs(args.download_page, raw_root)
        total_mb = sum(x["size_mb"] for x in items)
        print(f"\nDownloaded {len(items)} PDFs ({total_mb:.2f} MB total)")
        return 0

    if args.list_drive:
        entries = extract_drive_entries_with_labels(args.list_drive)
        print(f"Found {len(entries)} unique Drive files on {args.list_drive}:")
        for idx, (fid, label) in enumerate(entries, start=1):
            print(f"{idx:02d}. {fid} :: {label}")
        return 0

    targets = args.url or SEED_PAGES
    res = discover_links(targets)
    for url, info in res.items():
        if "error" in info:
            print(f"[ERR] {url}: {info['error']}")
            continue
        print(f"=== {url} (HTTP {info['status']}) pdfs={len(info['pdfs'])} drives={len(info['drives'])} subpages={len(info['subpages'])} ===")
        for p in info["pdfs"][:20]:
            print("  PDF:", p)
        for d in info["drives"][:10]:
            print("  DRV:", d)
        for s in info["subpages"][:8]:
            print("  SUB:", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
