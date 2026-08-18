import os
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from src.schemas.audit import AuditReport

# 1. Load GROQ_API_KEY from .env
load_dotenv()

# 2. State Definition
class AuditState(TypedDict):
    raw_text: str
    extracted_report: Optional[AuditReport]
    validation_errors: List[str]
    status: str

# 3. Initialize Groq with GPT-OSS 120B (Native Tool Calling)
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)
structured_llm = llm.with_structured_output(AuditReport)

def extractor_node(state: AuditState) -> dict:
    print("\n--- [Node 1: Extractor] Parsing Raw Text with Groq ---")
    prompt = f"Extract the audit report metrics from the following text:\n\n{state['raw_text']}"
    
    try:
        report: AuditReport = structured_llm.invoke(prompt)
        print("✅ Extraction Successful!")
        return {"extracted_report": report, "status": "EXTRACTED"}
    except Exception as e:
        print(f"\n❌ [ERROR IN NODE 1]: {e}\n")
        return {
            "validation_errors": [f"Extraction Error: {str(e)}"],
            "status": "EXTRACTION_FAILED"
        }

def validator_node(state: AuditState) -> dict:
    print("--- [Node 2: Validator] Running Financial Business Rules ---")
    report = state.get("extracted_report")
    
    if not report:
        return {
            "validation_errors": ["Validation Skipped: Extraction produced no report."],
            "status": "FAILED"
        }

    errors = []

    # 1. Math Sanity Check: Revenue vs EBITDA
    revenue = None
    ebitda = None
    for metric in report.metrics:
        name = metric.metric_name.lower()
        if "revenue" in name:
            revenue = metric.value
        elif "ebitda" in name:
            ebitda = metric.value

    if revenue is not None and ebitda is not None:
        if ebitda > revenue:
            errors.append(f"Financial Anomaly: EBITDA (${ebitda}M) cannot exceed Total Revenue (${revenue}M).")

    # 2. Compliance Flag Check
    if not report.compliance_flag:
        errors.append("Compliance Risk: Report indicates regulatory non-compliance.")

    # 3. Missing Metrics Warning
    if len(report.metrics) == 0:
        errors.append("Data Warning: No numeric financial metrics were extracted.")

    # Determine final status
    if errors:
        print(f"⚠️ Validation Flags Raised: {errors}")
        return {"validation_errors": errors, "status": "FLAGGED"}

    print("✅ All Financial & Compliance Rules Passed!")
    return {"validation_errors": [], "status": "PASSED"}

# 4. Graph Assembly
builder = StateGraph(AuditState)
builder.add_node("extract", extractor_node)
builder.add_node("validate", validator_node)

builder.add_edge(START, "extract")
builder.add_edge("extract", "validate")
builder.add_edge("validate", END)

graph = builder.compile()

# 5. Local Test Invocation
if __name__ == "__main__":
    sample_text = """
    Acme Corp Q3 2025 Financial Overview:
    Total revenue reached $14.5 million USD, with EBITDA recorded at $3.2 million USD. 
    All OSFI and FINTRAC regulatory guidelines were met.
    Summary: Financial state is strong and fully compliant.
    """

    initial_state = {
        "raw_text": sample_text,
        "extracted_report": None,
        "validation_errors": [],
        "status": "STARTING"
    }

    result = graph.invoke(initial_state)

    print("\n================ FINAL RESULTS ================")
    print(f"Final Status: {result['status']}")
    if result["extracted_report"]:
        print("\nExtracted Data (Pydantic Output):")
        print(result["extracted_report"].model_dump_json(indent=2))