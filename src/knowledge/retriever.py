import re

from neo4j import Driver
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.llm import OpenAILLM

from src.config import settings
from src.embeddings.factory import get_embedder

# Strip OpenAI's internal 【n†...】 file-citation annotations that leak through
# even when the prompt requests [n] format.
_OPENAI_CITATION_RE = re.compile(r'【\d+†[^】]*】')

_GROUNDING_RULE = (
    "IMPORTANT: Base your answer strictly on facts that appear in the provided context. "
    "Every claim in your answer must be traceable to the context text. "
    "If the context contains no relevant information for the question, "
    "respond with only this sentence: "
    "'Maaf, maklumat yang diperlukan tidak terdapat dalam konteks yang diberikan.'\n\n"
    "FORMATTING: Before giving a detailed answer, first ask the user in one short bilingual sentence "
    "what format they prefer — for example: "
    "'Adakah anda mahu jawapan dalam bentuk perenggan atau senarai? / "
    "Would you prefer the answer in paragraphs or as a list?' "
    "Then give a 1–2 sentence summary of what you found. "
    "Wait for the user to reply with their format preference before giving the full answer. "
    "Exception: if the user's question already specifies a format "
    "(e.g. 'list', 'senarai', 'jadual', 'table', 'bullet', 'points', 'ringkasan'), "
    "skip the format question and answer directly in that format.\n\n"
    "CITATIONS: Use only plain square-bracket numbers like [1] or [4] for inline citations. "
    "NEVER use 【】 brackets, dagger symbols (†), or line-range suffixes like L1-L4. "
    "Those formats are forbidden.\n\n"
)


def _build_template(works_cited: str = "") -> RagTemplate:
    if works_cited:
        instructions = (
            "Answer the user question using the provided context.\n\n"
            "The source text already contains inline citation numbers embedded directly "
            "in the content (e.g. 'berkembang sebanyak 8.7%4' or 'berstruktur1'). "
            "When you use a fact from the context, preserve those original inline numbers "
            "as superscript-style references like [4] or [1] immediately after the fact.\n\n"
            "At the end of your answer, output a '## References' section and list only "
            "the citation numbers you actually used, resolved using the Works Cited list below.\n\n"
            "Works Cited:\n"
            + works_cited
        )
    else:
        instructions = (
            "Answer the user question using the provided context. "
            "When you use information from the context, cite the source document "
            "inline as a superscript number like [1], and list the full document "
            "names as numbered references at the end under a '## References' heading. "
            "Only list documents you actually cited."
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


def build_rag(driver: Driver, works_cited: str = "") -> GraphRAG:
    return GraphRAG(llm=_build_llm(), retriever=build_retriever(driver), prompt_template=_build_template(works_cited))


def build_filtered_rag(driver: Driver, doc_filter: str, works_cited: str = "") -> GraphRAG:
    return GraphRAG(llm=_build_llm(), retriever=build_filtered_retriever(driver, doc_filter), prompt_template=_build_template(works_cited))


def search(rag: GraphRAG, question: str, top_k: int = 5) -> str:
    result = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
    )
    return _OPENAI_CITATION_RE.sub('', result.answer)
