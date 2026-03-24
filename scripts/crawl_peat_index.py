"""
Crawl https://expulsia.com/health/peat-index to extract all linked articles and PDFs
into the resources/ folder as markdown files prefixed with 'peat_'.
"""

import asyncio
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urljoin

import fitz
import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

BASE_URL = "https://expulsia.com/health/peat-index"
INDEX_URL = BASE_URL
RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"

SKIP_FILENAMES = {"MEGA Master Ray Newsletter.pdf"}

HTML_SEMAPHORE_LIMIT = 5
PDF_SEMAPHORE_LIMIT = 3
REQUEST_DELAY = 0.3  # seconds between requests to be polite

MIN_PDF_TEXT_LENGTH = 100  # minimum chars to consider a PDF as text-extractable


def slugify(name: str) -> str:
    """Convert a filename into a clean slug for the output markdown file."""
    name = unquote(name)
    name = Path(name).stem
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def derive_output_name(url: str) -> str:
    """Derive the peat_*.md output filename from a URL."""
    filename = url.rsplit("/", 1)[-1]
    slug = slugify(filename)

    # Normalize newsletter PDFs: e.g. "january_2018_ray_peat_s_newsletter" -> "peat_newsletter_2018_january"
    newsletter_match = re.match(
        r"^(?:q_\d+_)?(\w+)_(\d{4})_ray_peat_?s?_?new[sz]?letter.*$", slug
    )
    if newsletter_match:
        month_or_quarter = newsletter_match.group(1)
        year = newsletter_match.group(2)
        # Check if it's a quarterly format like "q_1_2022"
        q_match = re.match(r"^q_(\d+)_(\d{4})_ray_peat", slug)
        if q_match:
            quarter = q_match.group(1)
            year = q_match.group(2)
            return f"peat_newsletter_{year}_q{quarter}.md"
        return f"peat_newsletter_{year}_{month_or_quarter}.md"

    return f"peat_{slug}.md"


async def fetch_index(client: httpx.AsyncClient) -> tuple[list[str], list[str]]:
    """Fetch the index page and return lists of (html_urls, pdf_urls)."""
    resp = await client.get(INDEX_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    html_urls = []
    pdf_urls = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(INDEX_URL, href)

        # Only follow links under the peat-index path
        if "/peat-index/" not in full_url:
            continue

        filename = full_url.rsplit("/", 1)[-1]
        if filename in SKIP_FILENAMES:
            print(f"  SKIP (mega archive): {filename}")
            continue

        if full_url.lower().endswith(".pdf"):
            pdf_urls.append(full_url)
        elif full_url.lower().endswith(".html"):
            html_urls.append(full_url)

    # Deduplicate while preserving order
    html_urls = list(dict.fromkeys(html_urls))
    pdf_urls = list(dict.fromkeys(pdf_urls))

    return html_urls, pdf_urls


async def process_html(
    client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore
) -> tuple[str, str]:
    """Fetch an HTML article and convert to markdown. Returns (status, output_path_or_message)."""
    output_name = derive_output_name(url)
    output_path = RESOURCES_DIR / output_name

    async with semaphore:
        await asyncio.sleep(REQUEST_DELAY)
        try:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return "error", f"{url} -> {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        title = title_tag.string.strip()

    # Remove unwanted elements
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "iframe"]):
        tag.decompose()

    # Try to find main content area
    content = soup.find("main") or soup.find("article") or soup.find("body")
    if not content:
        return "error", f"{url} -> no content found"

    # Convert to markdown
    markdown_text = md(str(content), heading_style="ATX", strip=["img"])
    markdown_text = markdown_text.strip()

    if not markdown_text or len(markdown_text) < 50:
        return "error", f"{url} -> content too short ({len(markdown_text)} chars)"

    # Prepend title if not already in content
    if title and not markdown_text.startswith(f"# {title}"):
        markdown_text = f"# {title}\n\n{markdown_text}"

    # Clean up excessive blank lines
    markdown_text = re.sub(r"\n{4,}", "\n\n\n", markdown_text)

    output_path.write_text(markdown_text, encoding="utf-8")
    return "ok", str(output_path)


async def process_pdf(
    client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore
) -> tuple[str, str]:
    """Download a PDF and extract text to markdown. Returns (status, output_path_or_message)."""
    output_name = derive_output_name(url)
    output_path = RESOURCES_DIR / output_name

    async with semaphore:
        await asyncio.sleep(REQUEST_DELAY)
        try:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return "error", f"{url} -> {e}"

    # Write to temp file and extract with PyMuPDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(resp.content)
        tmp.flush()

        try:
            doc = fitz.open(tmp.name)
        except Exception as e:
            return "error", f"{url} -> failed to open PDF: {e}"

        pages_text = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text.strip())
        doc.close()

    full_text = "\n\n---\n\n".join(pages_text)

    if len(full_text.strip()) < MIN_PDF_TEXT_LENGTH:
        return "flagged", f"{url} -> text not recognizable ({len(full_text.strip())} chars extracted)"

    # Derive a title from the filename
    filename = unquote(url.rsplit("/", 1)[-1])
    title = Path(filename).stem.replace("-", " ").replace("_", " ")
    # Title-case it
    title = title.title()

    markdown_text = f"# {title}\n\n{full_text}"
    markdown_text = re.sub(r"\n{4,}", "\n\n\n", markdown_text)

    output_path.write_text(markdown_text, encoding="utf-8")
    return "ok", str(output_path)


async def main():
    print(f"Crawling {INDEX_URL}...")
    print(f"Output directory: {RESOURCES_DIR}\n")

    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=60.0) as client:
        html_urls, pdf_urls = await fetch_index(client)
        print(f"Found {len(html_urls)} HTML pages and {len(pdf_urls)} PDFs\n")

        html_sem = asyncio.Semaphore(HTML_SEMAPHORE_LIMIT)
        pdf_sem = asyncio.Semaphore(PDF_SEMAPHORE_LIMIT)

        # Process HTML pages
        print("--- Processing HTML pages ---")
        html_tasks = [process_html(client, url, html_sem) for url in html_urls]
        html_results = await asyncio.gather(*html_tasks)

        html_ok = sum(1 for s, _ in html_results if s == "ok")
        html_err = [(s, m) for s, m in html_results if s == "error"]
        print(f"  OK: {html_ok}, Errors: {len(html_err)}")
        for _, msg in html_err:
            print(f"    ERROR: {msg}")

        # Process PDFs
        print("\n--- Processing PDFs ---")
        pdf_tasks = [process_pdf(client, url, pdf_sem) for url in pdf_urls]
        pdf_results = await asyncio.gather(*pdf_tasks)

        pdf_ok = sum(1 for s, _ in pdf_results if s == "ok")
        pdf_flagged = [(s, m) for s, m in pdf_results if s == "flagged"]
        pdf_err = [(s, m) for s, m in pdf_results if s == "error"]
        print(f"  OK: {pdf_ok}, Flagged (unreadable): {len(pdf_flagged)}, Errors: {len(pdf_err)}")
        for _, msg in pdf_flagged:
            print(f"    FLAGGED: {msg}")
        for _, msg in pdf_err:
            print(f"    ERROR: {msg}")

        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"HTML pages: {html_ok}/{len(html_urls)} extracted")
        print(f"PDFs:       {pdf_ok}/{len(pdf_urls)} extracted")
        if pdf_flagged:
            print(f"Flagged PDFs (text not recognizable): {len(pdf_flagged)}")
        if html_err or pdf_err:
            print(f"Errors: {len(html_err) + len(pdf_err)}")


if __name__ == "__main__":
    asyncio.run(main())
