import io
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from pypdf import PdfReader
from groq import Groq

# LangGraph Agent (Harry)
from src.agent.graph import graph
from src.schemas.audit import AuditReport

# Qdrant Vector Store (Rutvik)
from src.platform.vector_store import index_document, search_document

app = FastAPI(
    title="AuditIQ Agent & RAG API",
    description="Production API combining LangGraph Financial Audit Agent and Qdrant RAG Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class AuditTextRequest(BaseModel):
    raw_text: str = Field(..., example="Acme Corp Q3 2025: Total revenue reached $14.5M USD, EBITDA was $3.2M USD. All regulatory checks met.")

class AuditReportResponse(BaseModel):
    status: str
    extracted_report: Optional[AuditReport] = None
    validation_errors: List[str] = []

class RAGQueryRequest(BaseModel):
    document_id: str = Field(..., example="acme_q3_2025")
    question: str = Field(..., example="What was the total revenue and did they pass compliance?")

class RAGQueryResponse(BaseModel):
    status: str
    answer: str
    confidence: float

# --- Healthcheck ---
@app.get("/health")
def healthcheck():
    return {"status": "healthy", "service": "AuditIQ Platform"}

# --- LangGraph Audit Endpoints ---
@app.post("/api/v1/audit/extract", response_model=AuditReportResponse, tags=["Audit Agent"])
def audit_raw_text(request: AuditTextRequest):
    """Parses raw text and validates financial metrics using LangGraph."""
    initial_state = {
        "raw_text": request.raw_text,
        "extracted_report": None,
        "validation_errors": [],
        "status": "STARTING"
    }
    try:
        result = graph.invoke(initial_state)
        return AuditReportResponse(
            status=result.get("status", "UNKNOWN"),
            extracted_report=result.get("extracted_report"),
            validation_errors=result.get("validation_errors", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution error: {str(e)}")

@app.post("/api/v1/audit/upload", response_model=AuditReportResponse, tags=["Audit Agent"])
async def audit_pdf_upload(file: UploadFile = File(...)):
    """Uploads a PDF financial report and extracts structured audit metrics using LangGraph."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        extracted_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from PDF.")

        initial_state = {
            "raw_text": extracted_text,
            "extracted_report": None,
            "validation_errors": [],
            "status": "STARTING"
        }
        result = graph.invoke(initial_state)
        return AuditReportResponse(
            status=result.get("status", "UNKNOWN"),
            extracted_report=result.get("extracted_report"),
            validation_errors=result.get("validation_errors", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF audit error: {str(e)}")

# --- Qdrant RAG Endpoints ---
@app.post("/api/v1/documents/{document_id}/index", tags=["RAG Document Store"])
async def index_document_endpoint(document_id: str, file: UploadFile = File(...)):
    """Uploads and indexes a PDF document into Qdrant vector database."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        extracted_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from PDF.")

        chunk_count = index_document(document_id=document_id, text=extracted_text)
        return {
            "status": "indexed",
            "document_id": document_id,
            "chunks_indexed": chunk_count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document indexing error: {str(e)}")

@app.post("/api/v1/rag/qa", response_model=RAGQueryResponse, tags=["RAG Document Store"])
async def rag_document_qa(request: RAGQueryRequest):
    """Retrieves document context from Qdrant and generates an answer using Groq."""
    try:
        results = search_document(
            document_id=request.document_id,
            question=request.question,
            limit=5
        )

        if not results:
            return RAGQueryResponse(
                status="no_context",
                answer="No relevant information was found for this document.",
                confidence=0.0
            )

        context = "\n\n".join([r["text"] for r in results])
        avg_score = sum([r["score"] for r in results]) / len(results)
        confidence = max(0.0, min(1.0, avg_score))

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")

        client = Groq(api_key=groq_api_key)
        prompt = f"""You are an enterprise audit assistant.
Answer the user's question using ONLY the provided document context.
If the answer cannot be found in the context, say that the information is not available.

Document context:
{context}

Question:
{request.question}

Provide a concise and factual answer."""

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return RAGQueryResponse(
            status="success",
            answer=completion.choices[0].message.content,
            confidence=round(confidence, 3)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A execution error: {str(e)}")