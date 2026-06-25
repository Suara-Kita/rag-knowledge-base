# RAG Knowledge Base — Local Setup

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.12+ | |
| Neo4j | 5.x | Community or Enterprise |
| Google AI API key | — | For Gemini embeddings |
| OpenRouter API key | — | For the answer LLM |
| Anthropic API key | — | For entity extraction at ingest |

---

## 1. Clone and create virtualenv

```bash
git clone https://github.com/Suara-Kita/rag-knowledge-base.git
cd rag-knowledge-base

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Configure environment

Copy the template and fill in your credentials:

```bash
cp .env.example .env   # or create .env manually
```

**.env**

```env
# Neo4j connection
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# LLM — answer generation (via OpenRouter)
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-...
LLM_MODEL=openai/gpt-oss-120b

# LLM — entity extraction at ingest (Anthropic)
ANTHROPIC_API_KEY=sk-ant-...
ENTITY_LLM_MODEL=claude-haiku-4-5-20251001

# Embeddings (Google Gemini)
EMBEDDING_MODEL=google/gemini-embedding-2
EMBEDDING_DIMS=3072

# Neo4j vector index
VECTOR_INDEX_NAME=chunk_embeddings

# Watch directories (comma-separated for multiple)
WATCH_DIR=./watch,./watch-johor-research
PROCESSED_DIR=./processed

# Server
MCP_PORT=8002
LOG_LEVEL=INFO
```

---

## 3. Start Neo4j

If using Neo4j Desktop, start your database and confirm it is listening on `bolt://localhost:7688`.

To restore the bundled dump (pre-loaded with Johor research):

```bash
# Stop Neo4j first, then:
neo4j-admin database load neo4j --from-path=./neo4j.dump --overwrite-destination
```

---

## 4. Ingest documents (optional if using the dump)

Place `.md` or `.pdf` files in `./watch` or `./watch-johor-research`.  
The server auto-ingests them on startup and polls for new files every 30 seconds.

To ingest a PDF manually before starting the server:

```bash
python -m src.pdf.ingest path/to/document.pdf
```

---

## 5. Run the server

```bash
python -m src.main
```

The server starts on **port 8002** and exposes:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/sse` | GET | MCP SSE connection (Claude Code / MCP clients) |
| `/messages/` | POST | MCP message transport |
| `/query` | POST | REST query endpoint for onn-ai chat |

---

## 6. Test the server

**REST query (quick test):**

```bash
curl -X POST http://localhost:8002/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Apa peluang pekerjaan di Johor?"}'
```

**List ingested documents:**

```bash
curl http://localhost:8002/query \
  -X POST -H 'Content-Type: application/json' \
  -d '{"question": "list documents"}'
```

**MCP SSE test (Python):**

```bash
python - << 'EOF'
import httpx, json, threading, queue

results = queue.Queue()
ready = threading.Event()
url_holder = []

def sse():
    with httpx.Client(timeout=30) as c:
        with c.stream("GET", "http://localhost:8002/sse",
                      headers={"Accept": "text/event-stream"}) as r:
            buf = {}
            for line in r.iter_lines():
                if line.startswith("event:"): buf["event"] = line[6:].strip()
                elif line.startswith("data:"): buf["data"] = line[5:].strip()
                elif line == "" and buf:
                    if buf.get("event") == "endpoint":
                        url_holder.append(f"http://localhost:8002{buf['data']}")
                        ready.set()
                    elif buf.get("event") == "message":
                        results.put(json.loads(buf["data"]))
                    buf = {}

import threading
t = threading.Thread(target=sse, daemon=True); t.start()
ready.wait(5)
msg_url = url_holder[0]

httpx.post(msg_url, json={"jsonrpc":"2.0","id":1,"method":"initialize",
    "params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}})
resp = results.get(timeout=5)
print("Server:", resp["result"]["serverInfo"])

httpx.post(msg_url, json={"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})
resp = results.get(timeout=5)
print("Tools:", [t["name"] for t in resp["result"]["tools"]])
EOF
```

---

## 7. Connect to Claude Code (MCP)

Ensure `.mcp.json` at the project root contains:

```json
{
  "mcpServers": {
    "suara-kita-knowledge": {
      "type": "sse",
      "url": "http://localhost:8002/sse"
    }
  }
}
```

Then restart Claude Code — it will connect to the running server automatically.

---

## 8. Run tests

```bash
# Unit tests only
pytest tests/unit

# Integration tests (requires running Neo4j + server)
pytest tests/integration -m integration
```

---

## Available MCP tools

| Tool | Description |
|------|-------------|
| `query_knowledge` | Search all ingested research documents |
| `query_johor_economy` | Filtered search — Johor/Malaysia macroeconomics only |
| `list_documents` | List all ingested documents with ingest timestamps |
