"""FastAPI REST API for the hotel RAG bot."""

import uuid
import time
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.rag_pipeline import RAGPipeline
from src.conversation import ConversationManager
from src.logger import logger

app = FastAPI(
    title="Hotel RAG Bot API",
    description="Grounded, highly robust hotel concierge chatbot powered by LangGraph + FAISS + Groq Failover",
    version="1.2.0"
)

# Custom CORS setup for modern web architecture
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dynamic request tracking logger
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"API Request | {request.method} {request.url.path} | "
        f"Status: {response.status_code} | Duration: {duration:.4f}s"
    )
    return response

# Pipeline instances
pipeline = RAGPipeline()
conv_manager = ConversationManager()


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        None, description="Unique conversation session UUID. If null, a new session is generated."
    )
    message: str = Field(
        ..., min_length=1, description="Raw query message sent by the hotel guest."
    )


class RetrievedDocSchema(BaseModel):
    id: str
    title: str
    score: float


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="UUID of the chat session.")
    response: str = Field(..., description="Grounded bot response in guest language.")
    intent: str = Field(..., description="Classified intent.")
    language: str = Field(..., description="Detected language.")
    guardrail_triggered: bool = Field(..., description="True if safety guardrails blocked the output.")
    retrieved_docs: list[RetrievedDocSchema] = Field([], description="Vector search context matches.")
    failover_active: bool = Field(False, description="True if the request executed via Groq failover.")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    user_msg = request.message.strip()

    if not user_msg:
        logger.warning("Rejecting request: Empty message string received.")
        raise HTTPException(status_code=400, detail="Message content cannot be blank.")

    try:
        # Load conversation sliding-window context
        history = conv_manager.get_history(session_id)
        
        # Run the RAG pipeline
        result = pipeline.run(user_msg, history)

        # Update in-memory session memory
        conv_manager.add_turn(session_id, "user", user_msg)
        conv_manager.add_turn(session_id, "assistant", result["response"])

        return ChatResponse(
            session_id=session_id,
            response=result["response"],
            intent=result["intent"],
            language=result["language"],
            guardrail_triggered=result["guardrail_triggered"],
            retrieved_docs=result["retrieved_docs"],
            failover_active=result.get("failover_active", False)
        )
    except Exception as e:
        logger.error(f"Internal Pipeline Error inside POST /chat: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while compiling your message. Please try again."
        )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    user_msg = request.message.strip()

    if not user_msg:
        logger.warning("Rejecting request: Empty message string received.")
        raise HTTPException(status_code=400, detail="Message content cannot be blank.")

    try:
        # Load conversation sliding-window context
        history = conv_manager.get_history(session_id)
        
        def event_generator():
            full_text = ""
            try:
                # Run the pipeline stream generator
                for chunk in pipeline.run_stream(user_msg, history):
                    if chunk["type"] == "token":
                        full_text += chunk["text"]
                    elif chunk["type"] == "replacement":
                        full_text = chunk["text"]
                    
                    # Yield as Newline Delimited JSON (NDJSON)
                    yield json.dumps(chunk) + "\n"
                
                # Commit clean complete turns to memory
                conv_manager.add_turn(session_id, "user", user_msg)
                conv_manager.add_turn(session_id, "assistant", full_text)
                
            except Exception as e:
                logger.error(f"Error in backend event generator: {e}")
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")
    except Exception as e:
        logger.error(f"Internal Pipeline Error inside POST /chat/stream: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while compiling your message stream. Please try again."
        )



@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    conv_manager.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared successfully."}


@app.get("/health")
async def health():
    status = "ok"
    details = {}
    
    # 1. Gemini check
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        details["gemini_api"] = "Key missing"
        status = "degraded"
    else:
        details["gemini_api"] = "healthy"

    # 2. Groq check
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key == "your_groq_api_key_here":
        details["groq_api"] = "Key missing (Failover Offline)"
    else:
        details["groq_api"] = "healthy (Failover Active)"

    # 3. Vector DB check
    # Check if retriever has self.kb
    from src.graph import retriever
    if retriever.index is None:
        status = "degraded"
        details["vector_index"] = "FAISS index missing"
    else:
        details["vector_index"] = "healthy"
        
    return {
        "status": status,
        "details": details,
        "timestamp": time.time()
    }
