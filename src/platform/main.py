import io
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq

from src.schemas.audit import AuditRequest, AuditResponse
from src.platform.vector_store import (
    index_document,
    search_document,
)


app = FastAPI(
    title="AuditIQ Agent API",
    description="AuditIQ RAG and audit intelligence platform",
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "service": "AuditIQ Platform",
    }


@app.post("/api/v1/documents/{document_id}/index")
async def index_document_endpoint(
    document_id: str,
    file: UploadFile = File(...),
):
    """
    Upload and index a PDF document into Qdrant.
    """

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    try:
        contents = await file.read()

        pdf_reader = PdfReader(
            io.BytesIO(contents)
        )

        extracted_text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                extracted_text += page_text + "\n"

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from PDF.",
            )

        chunk_count = index_document(
            document_id=document_id,
            text=extracted_text,
        )

        return {
            "status": "indexed",
            "document_id": document_id,
            "chunks_indexed": chunk_count,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document indexing error: {str(e)}",
        )


@app.post(
    "/api/v1/audit",
    response_model=AuditResponse,
)
async def run_audit(request: AuditRequest):
    """
    Retrieve relevant document context from Qdrant
    and generate an answer to the user's question.
    """

    try:
        results = search_document(
            document_id=request.document_id,
            question=request.question,
            limit=5,
        )

        if not results:
            return AuditResponse(
                status="no_context",
                answer="No relevant information was found for this document.",
                confidence=0.0,
            )

        context = "\n\n".join(
            result["text"]
            for result in results
        )

        average_score = sum(
            result["score"]
            for result in results
        ) / len(results)

        # Clamp similarity score to a sensible 0-1 range.
        confidence = max(
            0.0,
            min(1.0, average_score),
        )

        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY is not configured.",
            )

        client = Groq(
            api_key=groq_api_key
        )

        prompt = f"""
You are an enterprise audit assistant.

Answer the user's question using ONLY the provided
document context.

If the answer cannot be found in the context,
say that the information is not available.

Document context:
{context}

Question:
{request.question}

Provide a concise and factual answer.
"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        answer = completion.choices[0].message.content

        return AuditResponse(
            status="success",
            answer=answer,
            confidence=round(confidence, 3),
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Q&A execution error: {str(e)}",
        )