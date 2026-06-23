import re

from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks


class MarkdownSectionSplitter(TextSplitter):
    """Split markdown text on ## / ### headings so each section becomes one chunk.

    Sections smaller than min_chars are merged with the next section to avoid
    producing trivial empty-heading chunks (e.g. a lone '## ' line).
    Sections larger than max_chars are further split by the fixed-size fallback
    so no chunk exceeds the context window of the entity-extraction LLM.
    """

    def __init__(self, min_chars: int = 200, max_chars: int = 4000, overlap: int = 200) -> None:
        if overlap >= max_chars:
            raise ValueError(f"overlap ({overlap}) must be less than max_chars ({max_chars})")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap = overlap

    async def run(self, text: str) -> TextChunks:
        raw_sections = self._split_on_headings(text)
        merged = self._merge_small(raw_sections)
        final: list[str] = []
        for section in merged:
            if len(section) > self.max_chars:
                final.extend(self._fixed_split(section))
            else:
                final.append(section)

        chunks = [
            TextChunk(text=s.strip(), index=i)
            for i, s in enumerate(final)
            if s.strip()
        ]
        return TextChunks(chunks=chunks)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _split_on_headings(self, text: str) -> list[str]:
        """Split at ## / ### lines that carry actual heading text (not blank '## ' dividers)."""
        parts: list[str] = []
        current_lines: list[str] = []
        for line in text.splitlines(keepends=True):
            # Only treat as a heading boundary if there is text after the # markers
            if re.match(r"^#{2,3}\s+\S", line) and current_lines:
                parts.append("".join(current_lines))
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            parts.append("".join(current_lines))
        return parts

    def _merge_small(self, sections: list[str]) -> list[str]:
        """Merge sections shorter than min_chars into their successor."""
        merged: list[str] = []
        buffer = ""
        for section in sections:
            buffer += section
            if len(buffer) >= self.min_chars:
                merged.append(buffer)
                buffer = ""
        if buffer:
            if merged:
                merged[-1] += buffer
            else:
                merged.append(buffer)
        return merged

    def _fixed_split(self, text: str) -> list[str]:
        """Fallback: split a large section into overlapping fixed-size pieces."""
        step = self.max_chars - self.overlap
        parts: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            # avoid cutting mid-word
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            parts.append(text[start:end])
            start += step
        return parts
