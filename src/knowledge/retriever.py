import re
from typing import Any, Optional, Union

from neo4j import Driver
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage

from src.config import settings
from src.embeddings.factory import get_embedder

# Strip OpenAI's internal citation annotations.
# Bare 【n】 or range 【n‑m】/【n–m】 → [n] (keep first number); then strip dagger-form.
_OPENAI_CITATION_BARE_RE = re.compile(r'【(\d+)[‑–\-]?\d*】')
_OPENAI_CITATION_RE = re.compile(r'【\d+†[^】]*】')

# Fix citations that have a space before the sentence-ending period: "[1] ." → "[1]."
_CITATION_SPACE_PERIOD_RE = re.compile(r'(\[\d+(?:,\s*\d+)*\])\s+\.')

# Markdown patterns to strip from retrieved chunk text before it reaches the LLM.
# Source documents use *, -, ## etc. — if left in the context the model mirrors
# that formatting even when instructed to use prose.
_MD_HEADING = re.compile(r'^#{1,6}\s+\*{0,2}(.+?)\*{0,2}\s*$', re.MULTILINE)
_MD_BULLET = re.compile(r'^[ \t]*[-*]\s+', re.MULTILINE)
_MD_NUMBERED = re.compile(r'^[ \t]*\d+\.\s+', re.MULTILINE)
_MD_BOLD = re.compile(r'\*{1,3}([^*]+?)\*{1,3}')
_MD_EXTRA_BLANK = re.compile(r'\n{3,}')



def _strip_markdown(text: str) -> str:
    """Remove markdown formatting from a chunk so the LLM sees plain prose."""
    text = _MD_HEADING.sub(r'\1', text)       # ## Heading → Heading
    text = _MD_BULLET.sub('', text)            # - / * list markers removed
    text = _MD_NUMBERED.sub('', text)          # 1. 2. 3. markers removed
    text = _MD_BOLD.sub(r'\1', text)           # **bold** / *italic* → plain
    text = _MD_EXTRA_BLANK.sub('\n\n', text)   # collapse excess blank lines
    return text.strip()


_REF_SECTION_RE = re.compile(r'\n(?=## (?:References|Rujukan)\s*(?:\n|$))', re.IGNORECASE)
_MD_EXTENSION_RE = re.compile(r'(\.md)(?=\b|\s|$|\))', re.IGNORECASE)


def _strip_md_extensions(text: str) -> str:
    """Remove .md file extensions from reference entries so they read as document titles."""
    parts = _REF_SECTION_RE.split(text, maxsplit=1)
    if len(parts) == 1:
        return text
    return parts[0] + "\n" + _MD_EXTENSION_RE.sub('', parts[1])


_REF_ENTRY_RE = re.compile(r'^\[(\d+(?:,\s*\d+)*)\]\s+(.+)$', re.MULTILINE)


def _dedup_references(text: str) -> str:
    """Collapse repeated reference entries that point to the same document title.

    E.g. [1] Doc X and [5] Doc X → [1, 5] Doc X
    """
    parts = _REF_SECTION_RE.split(text, maxsplit=1)
    if len(parts) == 1:
        return text

    body, refs_block = parts[0], parts[1]

    doc_to_nums: dict[str, list[str]] = {}
    order: list[str] = []
    for m in _REF_ENTRY_RE.finditer(refs_block):
        nums_str, doc = m.group(1), m.group(2).strip()
        nums = [n.strip() for n in nums_str.split(',')]
        if doc not in doc_to_nums:
            doc_to_nums[doc] = []
            order.append(doc)
        doc_to_nums[doc].extend(nums)

    if not order:
        return text

    heading_match = re.match(r'(## (?:References|Rujukan)\s*\n)', refs_block, re.IGNORECASE)
    heading = heading_match.group(1) if heading_match else "## Rujukan\n"

    lines = [heading]
    for doc in order:
        nums = sorted(set(doc_to_nums[doc]), key=lambda x: int(x))
        lines.append(f"[{', '.join(nums)}] {doc}")

    return body + "\n" + "\n".join(lines)



class ProseGraphRAG(GraphRAG):
    """GraphRAG subclass that strips markdown from retrieved chunks before LLM sees them.

    This prevents the model from mirroring bullet/heading structure that appears
    in source documents, regardless of what the system prompt instructs.
    Calls the retriever and LLM each exactly once.
    """

    def search(
        self,
        query_text: str = "",
        message_history: Optional[Union[list[LLMMessage], MessageHistory]] = None,
        examples: str = "",
        retriever_config: Optional[dict[str, Any]] = None,
        return_context: Optional[bool] = None,
        response_fallback: Optional[str] = None,
    ) -> RagResultModel:
        if isinstance(message_history, MessageHistory):
            message_history = message_history.messages

        retrieval_query = self._build_query(query_text, message_history)
        retriever_result = self.retriever.search(
            query_text=retrieval_query, **(retriever_config or {})
        )

        if not retriever_result.items and response_fallback is not None:
            return RagResultModel(answer=response_fallback)

        clean_context = "\n\n".join(
            _strip_markdown(item.content) for item in retriever_result.items
        )
        prompt = self.prompt_template.format(
            query_text=query_text,
            context=clean_context,
            examples=examples,
        )
        llm_response = self.llm.invoke(
            input=prompt,
            message_history=message_history,
            system_instruction=self.prompt_template.system_instructions,
        )
        return RagResultModel(answer=llm_response.content)

_GROUNDING_RULE = (
    "IMPORTANT: Base your answer strictly on facts that appear in the provided context. "
    "Every claim must be traceable to the context text. "
    "If the context contains no relevant information, respond with only: "
    "'Maaf, maklumat yang diperlukan tidak terdapat dalam konteks yang diberikan.' "
    "(if the question was in Bahasa Melayu) or "
    "'Sorry, the required information is not available in the provided context.' "
    "(if the question was in English).\n\n"

    "LANGUAGE — MANDATORY: Detect the language of the user's question and reply exclusively "
    "in that same language. Translate facts from context as needed. Never mix languages.\n\n"

    "OUTPUT FORMATTING & STYLE — MANDATORY:\n"
    "Your goal is highly structured, scannable, visually appealing responses. "
    "Avoid dense walls of text.\n\n"

    "1. TYPOGRAPHY HIERARCHY\n"
    "   - Use ## headers to separate macro-topics into distinct modules.\n"
    "   - Use --- horizontal rules to separate major shifts in context.\n\n"

    "2. SCANNABILITY\n"
    "   - Never write body paragraphs longer than 3 lines.\n"
    "   - Bold **key phrases**, **metrics**, **monetary values**, and **critical dates** "
    "to guide the reader's eye. Do NOT bold entire sentences.\n\n"

    "3. STRUCTURED DATA LAYOUTS\n"
    "   - Bullet points: use for comparative data, lists of initiatives, or distinct items.\n"
    "   - Tables: whenever comparing 3 or more numerical metrics, tiers, or categories "
    "(e.g. salary brackets, project counts, benchmarks), format into a clean Markdown table.\n"
    "   - Blockquotes: use > for critical highlights, core objectives, or summary insights.\n\n"

    "4. CITATIONS\n"
    "   Use only plain square-bracket numbers like [1] or [4] inline. "
    "NEVER use 【】 brackets, dagger symbols (†), or line-range suffixes like L1-L4.\n\n"
)


def _build_template(works_cited: str = "") -> RagTemplate:
    _ref_heading_rule = (
        "At the end of your answer, output a references section using the EXACT heading below — "
        "choose based on the language of your answer:\n"
        "  - If answering in Bahasa Melayu → heading must be exactly: ## Rujukan\n"
        "  - If answering in English → heading must be exactly: ## References\n"
        "NEVER write '## References' when your answer is in Bahasa Melayu. "
        "NEVER write '## Rujukan' when your answer is in English. "
        "Group all citation numbers that resolve to the same document under a single entry — "
        "do not repeat the same document name more than once. "
        "Format: [n, m, ...] Document name. "
        "Only list documents you actually cited.\n\n"
    )
    if works_cited:
        instructions = (
            "Answer the user question using the provided context.\n\n"
            "The source text already contains inline citation numbers embedded directly "
            "in the content (e.g. 'berkembang sebanyak 8.7%4' or 'berstruktur1'). "
            "When you use a fact from the context, preserve those original inline numbers "
            "as superscript-style references like [4] or [1] immediately after the fact.\n\n"
            + _ref_heading_rule
            + "Works Cited:\n"
            + works_cited
        )
    else:
        instructions = (
            "Answer the user question using the provided context. "
            "When you use information from the context, cite the source document "
            "inline as a superscript number like [1], and list the full document "
            "names as numbered references at the end.\n\n"
            + _ref_heading_rule
        )
    return RagTemplate(system_instructions=_GROUNDING_RULE + instructions)


_retrieval_query = """
OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(d:Document)
OPTIONAL MATCH (e:Entity)-[:FROM_CHUNK]->(node)
RETURN node.text AS text,
       d.path AS document,
       collect(DISTINCT e.name) AS entities,
       score
"""

# Filtered to chunks from documents whose path contains a given keyword.
_filtered_retrieval_query = """
MATCH (node)-[:FROM_DOCUMENT]->(d:Document)
WHERE d.path CONTAINS $doc_filter
OPTIONAL MATCH (e:Entity)-[:FROM_CHUNK]->(node)
RETURN node.text AS text,
       d.path AS document,
       collect(DISTINCT e.name) AS entities,
       score
"""


def build_retriever(driver: Driver) -> VectorCypherRetriever:
    embedder = get_embedder()
    return VectorCypherRetriever(
        driver=driver,
        index_name=settings.vector_index_name,
        embedder=embedder,
        retrieval_query=_retrieval_query,
        neo4j_database=settings.neo4j_database,
    )


def build_filtered_retriever(driver: Driver, doc_filter: str) -> VectorCypherRetriever:
    embedder = get_embedder()
    # doc_filter is a controlled constant — safe to embed directly
    query = _filtered_retrieval_query.replace("$doc_filter", f'"{doc_filter}"')
    return VectorCypherRetriever(
        driver=driver,
        index_name=settings.vector_index_name,
        embedder=embedder,
        retrieval_query=query,
        neo4j_database=settings.neo4j_database,
    )


def _build_llm() -> OpenAILLM:  # noqa: F811
    return OpenAILLM(
        model_name=settings.llm_model,
        model_params={"temperature": 0},
    )


def build_rag(driver: Driver, works_cited: str = "") -> ProseGraphRAG:
    return ProseGraphRAG(llm=_build_llm(), retriever=build_retriever(driver), prompt_template=_build_template(works_cited))


def build_filtered_rag(driver: Driver, doc_filter: str, works_cited: str = "") -> ProseGraphRAG:
    return ProseGraphRAG(llm=_build_llm(), retriever=build_filtered_retriever(driver, doc_filter), prompt_template=_build_template(works_cited))


def search(rag: GraphRAG, question: str, top_k: int = 5) -> str:
    result = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
    )
    answer = _OPENAI_CITATION_BARE_RE.sub(r'[\1]', result.answer)  # 【n】/【n‑m】 → [n]
    answer = _OPENAI_CITATION_RE.sub('', answer)                   # strip dagger form
    answer = _CITATION_SPACE_PERIOD_RE.sub(r'\1.', answer)         # [1] . → [1].
    answer = _strip_md_extensions(answer)                           # strip .md from refs
    answer = _dedup_references(answer)                              # collapse duplicate docs
    return answer
