# 🛡️ AuditIQ: Autonomous Financial Audit Agent & Vector RAG Platform

AuditIQ is an enterprise-grade agentic financial intelligence system. It combines a stateful **LangGraph validation agent** for structured metric extraction and compliance checks with a high-throughput **Qdrant Vector RAG engine** for deep contextual document question answering.

---

## 🏛️ System Architecture

                ┌───────────────────────────────┐
                │      AuditIQ Streamlit UI     │
                │      (Port 8501 / Web App)    │
                └───────────────┬───────────────┘
                                │ HTTP REST
                                ▼
                ┌───────────────────────────────┐
                │      FastAPI Backend API      │
                │      (Port 8000 / Swagger)    │
                └───────┬───────────────┬───────┘
                        │               │
    ┌───────────────────┘               └───────────────────┐
    ▼                                                       ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│   LangGraph Agent Pipeline    │               │      Qdrant Vector Engine     │
│  ───────────────────────────  │               │  ───────────────────────────  │
│  1. In-Memory PDF Ingestion   │               │  1. Recursive Text Chunking   │
│  2. Structured Metric Extr.   │               │  2. FastEmbed (BGE-Small)     │
│  3. Business Rule Validator   │               │  3. Qdrant Embedded / Docker  │
│  4. PASSED / FLAGGED Output   │               │  4. Grounded Groq Q&A Synth   │
└───────────────────────────────┘               └───────────────────────────────┘

---

## ✨ Core Features

### 1. Autonomous Financial Audit Agent (LangGraph)
* **Direct PDF Ingestion:** Reads raw multi-page statements via `pypdf` in-memory without persistent disk clutter.
* **Deterministic Structured Extraction:** Uses Pydantic schema validation powered by Groq LLM inference.
* **Business Rule Validation Node:**
  * Ensures $\text{EBITDA} \le \text{Total Revenue}$.
  * Validates regulatory disclosures (OSFI, SEC, IFRS compliance).
  * Automatically tags reports as **`PASSED`** or **`FLAGGED`** with itemized violation descriptions.

### 2. Semantic Document RAG (Qdrant & FastEmbed)
* **Vector Indexing:** Splits financial documents into overlapping chunks and embeds them locally using `BAAI/bge-small-en-v1.5`.
* **Zero-Dependency Local Mode:** Runs embedded vector storage automatically (`./qdrant_db`) or scales to containerized Qdrant clusters.
* **Grounded Synthesis:** Queries vector collections with metadata filters and synthesizes factual answers with retrieval confidence metrics.

### 3. Full-Featured Streamlit UI
* Dual-tab layout for instantaneous audit execution and semantic document querying.
* Live status cards, metrics grids, compliance banners, and **CSV / JSON** report download actions.

---

## 🚀 Quick Start Guide

1. Clone & Set Up Virtual Environment

```bash
git clone [https://github.com/RutvikPathak/audit-iq.git](https://github.com/RutvikPathak/audit-iq.git)
cd audit-iq

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

2. Configure Environment Variables
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=http://localhost:6333 # Optional: leave blank for embedded local mode
QDRANT_COLLECTION=audit_documents

3. Launch the Backend API

python -m uvicorn src.platform.main:app --reload

#Interactive API Documentation (Swagger UI): http://127.0.0.1:8000/docs

4. Launch the Streamlit Dashboard

In a separate terminal tab:
python -m streamlit run src/ui/app.py

Dashboard URL: http://localhost:8501

🧪 Automated Testing
Execute the test suite covering clean reports, EBITDA anomalies, and compliance violations:

python -m pytest -v

📡 API Endpoints

Method        Endpoint                        Description

GET	          /health	                        Server health and status check
POST	        /api/v1/audit/upload	          Ingests PDF file, extracts financial schema, and runs LangGraph validation
POST	        /api/v1/audit/extract	          Audits raw text input passed via JSON
POST	        /api/v1/documents/{id}/index	  Chunks and embeds PDF document into Qdrant collection
POST	        /api/v1/rag/qa	                Semantic Q&A over indexed document collections

🛠️ Tech Stack
Frameworks: FastAPI, LangGraph, Streamlit, Pydantic v2

LLM & Embeddings: Groq (openai/gpt-oss-120b, openai/gpt-oss-20b), FastEmbed (BAAI/bge-small-en-v1.5)

Vector Database: Qdrant (Embedded local & Docker supported)

Testing: Pytest, HTTPX