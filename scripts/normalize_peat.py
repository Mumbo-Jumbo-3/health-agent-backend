"""
Normalize peat_*.md resource files using Gemini Flash.

Fixes OCR artifacts, broken line wrapping, misspellings, page numbers,
publication headers/footers, and ad blocks from PDF extraction.
Also normalizes HTML-sourced files for consistency.

Tracks progress in a .normalize_done file so it can be re-run safely
to pick up where it left off.
"""

import asyncio
import os
import sys
from pathlib import Path

from google import genai

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
DONE_FILE = RESOURCES_DIR / ".normalize_done"
CONCURRENCY = 10
REQUEST_DELAY = 1  # seconds between requests
MAX_RETRIES = 3
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """\
You are a text cleanup assistant. You will receive raw text extracted from PDFs \
and web pages about health, biology, and physiology by Dr. Ray Peat. \
Your job is to clean up the text and return well-formatted markdown.

Fix these issues:
1. **OCR misspellings**: Fix garbled characters (e.g. "8ttumulate" → "accumulate", \
"dilSolve" → "dissolve", "ofthe" → "of the", "ie" used as "is", \
"hirher" → "higher", "UII" → "us", "@(:t" → "effect"). Use context to determine \
the correct word.
2. **Broken lines**: Rejoin lines that were split mid-sentence or mid-word due to \
PDF column layout. Reflow into natural paragraphs.
3. **Page artifacts**: Remove page numbers, publication headers/footers \
(e.g. "Townsend Letter for Doctors August/September 1992"), subscription info, \
ad blocks, mailing addresses for the publication, and other non-content material.
4. **Stray characters**: Remove lone characters that are column separator artifacts \
(e.g. lines containing only "I" or "|").
5. **Formatting**: Use proper markdown — headers (#, ##), paragraphs separated by \
blank lines, lists where appropriate. Keep the H1 title at the top.

Rules:
- Do NOT change the meaning, add new content, or editorialize.
- Do NOT summarize or shorten — preserve ALL substantive content.
- Keep reference lists, footnotes, and citations intact (clean up formatting only).
- If the text is already clean, return it as-is with minimal changes.
- Return ONLY the cleaned markdown text, no preamble or explanation.
- Preserve author attributions like "Raymond Peat, Ph.D." at the end of articles.\
"""


def load_done_set() -> set[str]:
    """Load the set of already-normalized filenames."""
    if DONE_FILE.exists():
        return set(DONE_FILE.read_text().strip().splitlines())
    return set()


def mark_done(filename: str):
    """Append a filename to the done tracking file."""
    with DONE_FILE.open("a") as f:
        f.write(filename + "\n")


async def normalize_file(
    client: genai.Client,
    filepath: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    """Normalize a single file with retry logic. Returns (status, message)."""
    async with semaphore:
        content = filepath.read_text(encoding="utf-8")

        if len(content.strip()) < 50:
            return "skip", f"{filepath.name} (too short)"

        for attempt in range(1, MAX_RETRIES + 1):
            await asyncio.sleep(REQUEST_DELAY)
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=content,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=65536,
                    ),
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < MAX_RETRIES:
                    wait = 60 * attempt
                    print(f"    Rate limited on {filepath.name}, waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                    await asyncio.sleep(wait)
                    continue
                return "error", f"{filepath.name} -> {e}"

        if not response.text:
            return "error", f"{filepath.name} -> empty response"

        cleaned = response.text.strip()

        # Remove markdown code fences if the model wrapped its output
        if cleaned.startswith("```markdown"):
            cleaned = cleaned[len("```markdown") :].strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        filepath.write_text(cleaned + "\n", encoding="utf-8")
        mark_done(filepath.name)
        return "ok", filepath.name


async def main():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    all_files = sorted(RESOURCES_DIR.glob("peat_*.md"))
    done_set = load_done_set()
    files = [f for f in all_files if f.name not in done_set]

    print(f"Found {len(all_files)} peat_*.md files total, {len(done_set)} already done")
    print(f"Processing {len(files)} remaining files")
    print(f"Model: {MODEL}, Concurrency: {CONCURRENCY}, Delay: {REQUEST_DELAY}s\n")

    if not files:
        print("Nothing to do!")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [normalize_file(client, f, semaphore) for f in files]

    ok_count = 0
    err_count = 0
    skip_count = 0

    for coro in asyncio.as_completed(tasks):
        status, msg = await coro
        total = ok_count + err_count + skip_count + 1
        if status == "ok":
            ok_count += 1
            print(f"  [{total}/{len(files)}] OK: {msg}")
        elif status == "error":
            err_count += 1
            print(f"  [{total}/{len(files)}] ERROR: {msg}")
        else:
            skip_count += 1
            print(f"  [{total}/{len(files)}] SKIP: {msg}")

    print(f"\n=== SUMMARY ===")
    print(f"Normalized: {ok_count}/{len(files)}")
    print(f"Previously done: {len(done_set)}")
    if skip_count:
        print(f"Skipped: {skip_count}")
    if err_count:
        print(f"Errors: {err_count}")


if __name__ == "__main__":
    asyncio.run(main())
