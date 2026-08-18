import pytest
from src.agent.graph import graph

def test_clean_report_passes():
    """Test 1: Clean report with valid figures must return PASSED."""
    payload = {
        "raw_text": "Alpha Corp Q2 2025: Total revenue was $20.0M USD, with EBITDA at $4.5M USD. Full compliance with OSFI regulations confirmed.",
        "extracted_report": None,
        "validation_errors": [],
        "status": "STARTING"
    }
    result = graph.invoke(payload)
    
    assert result["status"] == "PASSED"
    assert len(result["validation_errors"]) == 0
    assert result["extracted_report"].company_name == "Alpha Corp"

def test_ebitda_exceeds_revenue_flagged():
    """Test 2: EBITDA higher than Revenue is mathematically impossible and must be FLAGGED."""
    payload = {
        "raw_text": "BadMath LLC Q1 2025: Total revenue was $2.0M USD, EBITDA was reported at $8.0M USD. Compliance checks passed.",
        "extracted_report": None,
        "validation_errors": [],
        "status": "STARTING"
    }
    result = graph.invoke(payload)
    
    assert result["status"] == "FLAGGED"
    assert any("EBITDA" in err for err in result["validation_errors"])

def test_compliance_failure_flagged():
    """Test 3: Regulatory non-compliance must trigger FLAGGED status."""
    payload = {
        "raw_text": "RiskCorp Q4 2024: Total revenue was $50.0M USD, EBITDA was $10.0M USD. The company failed OSFI regulatory audit standards.",
        "extracted_report": None,
        "validation_errors": [],
        "status": "STARTING"
    }
    result = graph.invoke(payload)
    
    assert result["status"] == "FLAGGED"
    assert any("regulatory" in err.lower() or "compliance" in err.lower() for err in result["validation_errors"])