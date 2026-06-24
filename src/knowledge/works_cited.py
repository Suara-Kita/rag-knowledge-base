import re
from pathlib import Path


def extract_works_cited(md_path: Path) -> str:
    """Return the Works Cited block from a markdown file as a plain numbered list.

    Looks for a heading containing 'Works cited' (case-insensitive) and collects
    every numbered entry that follows until the next heading or end of file.
    Returns an empty string if nothing is found.
    """
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    inside = False
    entries: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Detect Works Cited or References heading (any level of #, bold markers, etc.)
        if re.search(r"(works\s+cited|references)", stripped, re.IGNORECASE):
            inside = True
            continue

        if inside:
            # Stop at next markdown heading
            if stripped.startswith("#"):
                break
            # Skip base64 image data lines
            if stripped.startswith("[image") or stripped.startswith("data:image"):
                break
            # Collect numbered reference lines: start with a digit followed by dot/paren
            if re.match(r"^\d+[\.\)]", stripped):
                # Clean up escaped characters and extra whitespace
                clean = re.sub(r"\\(.)", r"\1", stripped)
                entries.append(clean)

    return "\n".join(entries)
