"""FastAPI REST API for the hotel RAG bot."""

import uuid
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.rag_pipeline import RAGPipeline
from src.conversation import ConversationManager
from src.logger import logger

app = FastAPI(
    title="Hotel RAG Bot API",
    description="Grounded, highly robust hotel concierge chatbot powered by Gemini + FAISS",
    version="1.1.0"
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
            retrieved_docs=result["retrieved_docs"]
        )
    except Exception as e:
        logger.error(f"Internal Pipeline Error inside POST /chat: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while compiling your message. Please try again."
        )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    conv_manager.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared successfully."}


@app.get("/health")
async def health():
    # Simple dependency verification
    status = "ok"
    details = {}
    
    if pipeline.model is None:
        status = "degraded"
        details["gemini_api"] = "Model failed to initialize. Check GEMINI_API_KEY."
    else:
        details["gemini_api"] = "healthy"

    if pipeline.retriever.index is None:
        status = "degraded"
        details["vector_index"] = "FAISS index missing. Run scripts/build_index.py."
    else:
        details["vector_index"] = "healthy"
        
    return {
        "status": status,
        "details": details,
        "timestamp": time.time()
    }
