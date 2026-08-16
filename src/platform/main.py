from fastapi import FastAPI

from src.schemas.audit import AuditRequest, AuditResponse

app = FastAPI()


@app.post("/api/v1/audit", response_model=AuditResponse)
def audit(request: AuditRequest):
    return AuditResponse(
        status="completed",
        answer="This is a mock audit response.",
        confidence=0.95,
    )