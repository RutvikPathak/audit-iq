from typing import List
from pydantic import BaseModel, Field


class FinancialMetric(BaseModel):
    metric_name: str = Field(..., description="Name of metric, e.g., Revenue")
    value: float = Field(..., description="Numeric value extracted")
    unit: str = Field(default="USD")


class AuditReport(BaseModel):
    company_name: str
    fiscal_quarter: str
    metrics: List[FinancialMetric]
    compliance_flag: bool
    summary: str


class AuditRequest(BaseModel):
    document_id: str
    question: str


class AuditResponse(BaseModel):
    status: str
    answer: str
    confidence: float