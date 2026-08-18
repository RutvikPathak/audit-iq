import io
from dotenv import load_dotenv

# Load environment variables (GROQ_API_KEY)
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from pypdf import PdfReader

from src.agent.graph import graph
from src.schemas.audit import AuditReport

# --- 1. Initialize App ---
app = FastAPI(
    title="AuditIQ Agent API",
    description="Production API for AuditIQ LangGraph extraction and validation engine",
    version="1.0.0"
)

# --- 2. Enable CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local dev / frontend integration
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Request & Response Schemas ---
class AuditRequest(BaseModel):
    raw_text: str = Field(
        ...,
        description="Raw financial report or transcript text to audit",
        example="Acme Corp Q3 2025: Total revenue reached $14.5M USD, EBITDA was $3.2M USD. All regulatory checks met."
    )

class AuditResponse(BaseModel):
    status: str
    extracted_report: Optional[AuditReport] = None
    validation_errors: List[str] = []

# --- 4. Healthcheck Endpoint ---
@app.get("/health")
def healthcheck():
    return {"status": "healthy", "service": "AuditIQ Platform"}

# --- 5. Raw Text Audit Endpoint ---
@app.post("/api/v1/audit", response_model=AuditResponse)
def audit_raw_text(request: AuditRequest):
    """Audits raw text passed directly via JSON payload."""
    initial_state = {
        "raw_text": request.raw_text,
        "extracted_report": None,
        "validation_errors": [],
        "status": "STARTING"
    }

    try:
        result = graph.invoke(initial_state)
        return AuditResponse(
            status=result.get("status", "UNKNOWN"),
            extracted_report=result.get("extracted_report"),
            validation_errors=result.get("validation_errors", [])
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution error: {str(e)}"
        )

# --- 6. PDF File Upload Audit Endpoint ---
@app.post("/api/v1/audit/upload", response_model=AuditResponse)
async def audit_pdf_upload(file: UploadFile = File(...)):
    """Uploads a PDF financial report, extracts text in memory, and runs the audit agent."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .pdf file.")

    try:
        # Read file contents into memory
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        extracted_text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from the uploaded PDF. It may be scanned or empty."
            )

        initial_state = {
            "raw_text": extracted_text,
            "extracted_report": None,
            "validation_errors": [],
            "status": "STARTING"
        }

        result = graph.invoke(initial_state)
        return AuditResponse(
            status=result.get("status", "UNKNOWN"),
            extracted_report=result.get("extracted_report"),
            validation_errors=result.get("validation_errors", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF ingestion/audit error: {str(e)}"
        )