import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any

from src.agent.graph import graph
from src.schemas.audit import AuditReport

# 1. Initialize FastAPI App
app = FastAPI(
    title="AuditIQ Agent API",
    description="REST API interface for AuditIQ LangGraph extraction and validation engine",
    version="1.0.0",
)

# 2. Request & Response Schemas
class AuditRequest(BaseModel):
    raw_text: str = Field(
        ..., 
        description="Raw financial report or transcript text to audit",
        example="Acme Corp Q3 2025: Total revenue reached $14.5M USD, EBITDA was $3.2M USD. All regulatory checks met."
    )

class AuditResponse(BaseModel):
    status: str
    extracted_report: Optional[AuditReport] = None
    validation_errors: list[str] = []

# 3. Healthcheck Endpoint
@app.get("/health")
def healthcheck():
    return {"status": "healthy", "service": "AuditIQ Agent Backend"}

# 4. Audit Execution Endpoint
@app.post("/api/v1/audit", response_model=AuditResponse)
async def run_audit(request: AuditRequest):
    """
    Receives raw text, runs it through LangGraph (Extractor -> Validator),
    and returns structured audit results.
    """
    initial_state = {
        "raw_text": request.raw_text,
        "extracted_report": None,
        "validation_errors": [],
        "status": "STARTING"
    }

    try:
        # Invoke your compiled LangGraph pipeline
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