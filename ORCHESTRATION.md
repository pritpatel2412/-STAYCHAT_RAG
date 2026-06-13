# StayChat RAG — System Orchestration Guide

## 1. Project Overview

**StayChat RAG** is a production-grade hotel concierge chatbot for *The Grand Hotel, Mumbai*.  
It uses a **stateful LangGraph workflow** to orchestrate a multi-step Retrieval-Augmented Generation (RAG) pipeline with dual-LLM failover, hybrid search, multi-layer guardrails, and real-time streaming.

| Layer | Technology |
|-------|-----------|
| **Orchestration** | LangGraph (StateGraph) |
| **Primary LLM** | Google Gemini 2.5 Flash |
| **Failover LLM** | Groq LLaMA-3.3 70B |
| **Embeddings** | Gemini Embedding v2 |
| **Vector Store** | FAISS (IndexFlatIP, cosine) |
| **Keyword Search** | Custom BM25 (zero-dependency) |
| **Rank Fusion** | Reciprocal Rank Fusion (RRF) |
| **API Server** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Session Memory** | In-memory sliding window (10 turns) |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND                          │
│   (app.py — Premium UI, telemetry dashboard, chat interface)       │
└─────────────┬──────────────────────────────────────┬────────────────┘
              │ HTTP (API Mode)                      │ Direct (Standalone Mode)
              ▼                                      ▼
┌─────────────────────────┐            ┌──────────────────────────────┐
│   FastAPI REST Server   │            │     RAGPipeline (Direct)     │
│   (api/main.py)         │            │   (src/rag_pipeline.py)      │
│   POST /chat            │            │   .run() / .run_stream()     │
│   POST /chat/stream     │            └──────────────┬───────────────┘
│   GET  /health          │                           │
│   DELETE /session/{id}  │                           │
└────────────┬────────────┘                           │
             │                                        │
             ▼                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│              COMPILED LANGGRAPH STATE WORKFLOW                      │
│                    (src/graph.py)                                   │
│                                                                     │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │   INPUT      │───▶│  LANGUAGE        │───▶│  INTENT          │   │
│  │  GUARDRAIL   │    │  DETECTOR        │    │  CLASSIFIER      │   │
│  └──────┬───────┘    └─────────────────┘    └────────┬─────────┘   │
│         │ (if unsafe → END)                          │             │
│         │                                            ▼             │
│         │                                  ┌──────────────────┐    │
│         │                                  │  HYBRID          │    │
│         │                                  │  RETRIEVER       │    │
│         │                                  │  (FAISS + BM25)  │    │
│         │                                  └────────┬─────────┘    │
│         │                                           ▼              │
│         │                                  ┌──────────────────┐    │
│         │                                  │  RESPONSE        │    │
│         │                                  │  GENERATOR       │    │
│         │                                  │  (Gemini→Groq)   │    │
│         │                                  └────────┬─────────┘    │
│         │                                           ▼              │
│         │                                  ┌──────────────────┐    │
│         │                                  │  OUTPUT          │    │
│         │                                  │  GUARDRAIL       │    │
│         │                                  └────────┬─────────┘    │
│         │                                           │              │
│         ▼                                           ▼              │
│       [END] ◄───────────────────────────────────  [END]            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. LangGraph State Machine — Detailed Node Walkthrough

The core orchestration is a **compiled LangGraph `StateGraph`** defined in `src/graph.py`.  
Every node reads from and writes to a shared **`AgentState`** typed dictionary:

```python
class AgentState(TypedDict):
    message: str                        # Raw guest input
    session_id: str                     # UUID for conversation tracking
    history: List[Dict[str, str]]       # Sliding-window conversation context
    language: str                       # Detected language: en | hi | hinglish
    intent: str                         # Classified intent category
    retrieved_docs: List[Dict[str, Any]]# Hybrid retrieval results
    response: str                       # Final generated text
    guardrail_triggered: bool           # Whether any guardrail fired
    guardrail_reason: str               # Why the guardrail fired
    failover_active: bool               # Whether Groq fallback was used
```

### 3.1 Node 1 — Input Guardrail (`input_guardrail_node`)

**File:** `src/input_guardrail.py`  
**Purpose:** Intercepts adversarial inputs *before* they enter the pipeline.

**Two-layer defense:**

| Layer | Method | Speed | Cost |
|-------|--------|-------|------|
| **Level 1** | Regex pattern matching against 8 known prompt-injection signatures | Instant | Free |
| **Level 2** | LLM-based classification (Gemini → Groq fallback) | ~1s | 1 API call |

**Level 1 — Local Signature Patterns:**
```
• "ignore all prior instructions"
• "ignore your guidelines"
• "system override"
• "bypass safety filters"
• "decode system prompt"
• "reveal instruction keys"
• "forget instructions"
• "new role is now"
```

**Level 2 — LLM Classification:**  
Only triggered when the input contains suspicious keywords (`instruction`, `guideline`, `system`, `override`, `bypass`, `prompt`, `forget`).  
The LLM is asked to classify the query as `SAFE` or `UNSAFE`.

**Routing Logic (Conditional Edge):**

```
IF guardrail_triggered == True:
    → Route directly to END (skip all downstream nodes)
ELSE:
    → Route to language_detector
```

This is implemented as a **conditional edge** in LangGraph:
```python
workflow.add_conditional_edges(
    "input_guardrail",
    route_after_input_guardrail,
    { END: END, "language_detector": "language_detector" }
)
```

---

### 3.2 Node 2 — Language Detector (`language_detector_node`)

**File:** `src/language_detector.py`  
**Purpose:** Detects whether the guest is speaking English, Hindi, or Hinglish.

**Four-layer detection cascade:**

```
Step 1: Unicode Range Check
    → Devanagari characters (U+0900–U+097F) detected?  →  "hi"

Step 2: Hinglish Keyword Markers
    → Matches from a curated set of 26 Romanized Hindi words?  →  "hinglish"
    → (e.g., "kya", "hai", "chahiye", "kitne", "batao")

Step 3: Pure ASCII Check
    → All characters < 128 and no Hinglish markers?  →  "en"

Step 4: Gemini LLM Fallback
    → Non-ASCII text that didn't match any heuristic?
    → Query Gemini for classification  →  "en" | "hi" | "hinglish"
```

**Design Decision:** The `"please"` keyword was intentionally excluded from Hinglish markers to prevent false positives on standard English messages.

---

### 3.3 Node 3 — Intent Classifier (`intent_classifier_node`)

**File:** `src/intent_classifier.py`  
**Purpose:** Classifies the guest's message into one of 5 intent categories.

| Intent | Description |
|--------|-------------|
| `booking_inquiry` | Reservations, check-in/out, cancellation, pricing, payments |
| `amenity_question` | Hotel facilities, spa, pool, gym, Wi-Fi, parking, dining |
| `complaint` | Dissatisfaction, reporting problems |
| `staff_command` | Action requests ("send towels", "need housekeeping") |
| `other` | Anything outside the above categories |

**Execution Flow:**
```
1. Send classification prompt to Gemini 2.5 Flash
2. Validate response is in VALID_INTENTS set
3. If invalid response → default to "other"
4. If Gemini API fails (rate-limit / error):
   a. Send same prompt to Groq LLaMA-3.3 70B
   b. Parse via substring matching against VALID_INTENTS
   c. Set failover_active = True
5. If both fail → default to "other"
```

---

### 3.4 Node 4 — Hybrid Retriever (`retriever_node`)

**File:** `src/retriever.py`  
**Purpose:** Retrieves the most relevant hotel knowledge base documents using a **hybrid search strategy**.

#### 3.4.1 Knowledge Base

**Source:** `data/hotel_kb.json` — A structured JSON file containing hotel information  
**FAISS Index:** `faiss_index/hotel.index` — Pre-built dense vector index  
**Metadata:** `faiss_index/hotel_meta.pkl` — Pickled KB entries for result enrichment

The index is built offline by `src/ingest.py` which:
1. Loads `hotel_kb.json`
2. Constructs rich text strings: `"Category: {cat}\nTitle: {title}\n{content}"`
3. Sends all texts in a **single batched** API call to `gemini-embedding-2`
4. L2-normalizes the vectors
5. Builds a `faiss.IndexFlatIP` index (inner product = cosine after normalization)
6. Saves index + metadata to disk

#### 3.4.2 Dual-Channel Search

**Channel A — FAISS Semantic Search (Dense Retrieval):**
```
1. Embed query using gemini-embedding-2 (task_type="retrieval_query")
2. L2-normalize the query vector
3. Search FAISS index for top-K nearest neighbors (inner product)
4. Return documents with similarity_score
```

**Channel B — BM25 Keyword Search (Sparse Retrieval):**
```
1. Tokenize query into lowercase word tokens
2. For each document in corpus, compute BM25 score:
   BM25(q, d) = Σ IDF(t) × [ tf(t,d) × (k1+1) ] / [ tf(t,d) + k1 × (1 - b + b × |d|/avgdl) ]
   Parameters: k1=1.5, b=0.75
3. Sort by score descending, return top-K with score > 0
```

#### 3.4.3 Reciprocal Rank Fusion (RRF)

The two result sets are combined using **RRF** with K=60:

```
For each document appearing in either result set:
    RRF_Score(d) = Σ  1 / (K + rank_i)
    where rank_i is the 1-based rank of document d in result set i

Final results = sort by RRF_Score descending, take top-K
```

**Why RRF?**  
- Semantic search captures meaning ("Where can I relax?" → finds spa/pool docs)
- BM25 captures exact keywords ("WiFi password" → exact keyword match)
- RRF gives a document appearing in *both* lists a higher combined score without needing score normalization

---

### 3.5 Node 5 — Response Generator (`response_generator_node`)

**Purpose:** Generates a grounded response using retrieved context.

**Pre-Generation Guardrail (Level 1):**
```
Before generating, check if the TOP retrieved document has similarity_score >= 0.40
If NOT:
    → Return a human handoff message (in the guest's language)
    → Set guardrail_triggered = True
    → Skip generation entirely
```

**Prompt Construction:**
```
System Prompt includes:
    1. Role definition (professional hotel concierge)
    2. 8 critical grounding rules (no price fabrication, no URL invention, etc.)
    3. Language instruction (respond in guest's detected language)
    4. Retrieved CONTEXT block (from hybrid search)
    5. Conversation HISTORY (last 6 turns)
    6. Classified INTENT
    7. Guest MESSAGE
```

**Dual-LLM Execution with Failover:**
```
TRY:
    → Generate via Gemini 2.5 Flash
CATCH (rate-limit, API error):
    → TRY:
        → Generate via Groq LLaMA-3.3 70B
        → Set failover_active = True
    CATCH:
        → Return human handoff message
        → Set guardrail_triggered = True
```

---

### 3.6 Node 6 — Output Guardrail (`output_guardrail_node`)

**File:** `src/guardrail.py`  
**Purpose:** Post-generation scan for hallucinations and policy violations.

**Bypass Logic:** If a guardrail was already triggered in a previous node (e.g., input guardrail or low-similarity), this node is skipped entirely.

**Level 2 — Hallucination Pattern Scanning:**

| Check | Pattern | Example Violation |
|-------|---------|------------------|
| **Price Fabrication** | Currency symbols (USD, EUR, $, €, ₹) followed by digits | "$500 per night" |
| **Cost Expressions** | "price is" / "costs" followed by numbers | "The price is Rs 5000" |
| **Fabricated Phones** | 10+ digit numbers not matching known KB numbers | "Call 9876543210" |
| **Unauthorized URLs** | Any URL not in whitelist (`grandhotel.com`, `google.com`, `maps.google`) | "Book at www.fakehotel.com" |

**If any violation is found:**
```
→ Replace the entire generated response with a human handoff message
→ Set guardrail_triggered = True
→ Log the specific violation reason
```

---

## 4. Edge Topology (Complete Graph Wiring)

```
ENTRY POINT
    │
    ▼
input_guardrail ──[CONDITIONAL]──► language_detector ──► intent_classifier ──► retriever ──► response_generator ──► output_guardrail ──► END
    │                                                                                                                                      
    │ (if unsafe)                                                                                                                          
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────► END
```

**Edge Definitions in Code:**
```python
workflow.set_entry_point("input_guardrail")

# Conditional: skip pipeline if input is adversarial
workflow.add_conditional_edges("input_guardrail", route_fn, {END, "language_detector"})

# Sequential flow
workflow.add_edge("language_detector", "intent_classifier")
workflow.add_edge("intent_classifier", "retriever")
workflow.add_edge("retriever", "response_generator")
workflow.add_edge("response_generator", "output_guardrail")
workflow.add_edge("output_guardrail", END)
```

---

## 5. Streaming Architecture

The `RAGPipeline.run_stream()` method (in `src/rag_pipeline.py`) provides **real-time token-by-token streaming** for the Streamlit UI.

### 5.1 Event Types

| Event Type | Payload | Purpose |
|-----------|---------|---------|
| `telemetry` | `{intent, language, retrieved_docs, failover_active}` | Sent immediately after retrieval — populates the sidebar telemetry panel |
| `telemetry_update` | `{failover_active: true}` | Sent if Gemini fails mid-stream and Groq takes over |
| `token` | `{text: "..."}` | Individual text chunk for typewriter animation |
| `replacement` | `{text: "..."}` | Output guardrail triggered — replace all streamed text with safe refusal |
| `error` | `{message: "..."}` | Pipeline error message |

### 5.2 Streaming Flow

```
1. Input Guardrail scan
   ├── UNSAFE → yield telemetry + refusal token → RETURN
   └── SAFE → continue

2. Language Detection (local heuristics)

3. Intent Classification (Gemini → Groq fallback)

4. Hybrid Retrieval (FAISS + BM25 + RRF)

5. ► YIELD telemetry event (populates sidebar instantly)

6. Pre-generation similarity check
   ├── FAIL → yield human handoff token → RETURN
   └── PASS → continue

7. Streaming Generation:
   ├── TRY Gemini streaming → yield tokens chunk-by-chunk
   │   └── ON FAILURE → yield telemetry_update → switch to Groq
   └── Groq streaming → yield tokens chunk-by-chunk

8. Post-generation Output Guardrail
   └── IF violations found → yield replacement event (overwrites all streamed text)
```

---

## 6. Dual-LLM Failover Strategy

The system implements **automatic failover** at every LLM-dependent stage:

```
┌─────────────────┐     ┌──────────────────────────────────┐     ┌──────────────────┐
│ Input Guardrail  │────▶│ Gemini 2.5 Flash (Primary)       │────▶│ Groq LLaMA-3.3   │
│ LLM Scan         │     │ - Safety classification          │     │ 70B (Fallback)    │
└─────────────────┘     │ - Intent classification          │     │                    │
                         │ - Language detection (fallback)   │     │ - Same prompts     │
                         │ - Response generation             │     │ - Same validation  │
                         │ - Streaming generation            │     │ - Streaming support│
                         └──────────────────────────────────┘     └──────────────────┘
                                        │                                  ▲
                                        │  On rate-limit / API error       │
                                        └──────────────────────────────────┘
```

**Failover triggers on:**
- HTTP 429 (rate limit exceeded)
- HTTP 5xx (server errors)
- Network timeouts
- Any unhandled API exception

**The Groq client** (`src/groq_client.py`) is a zero-dependency, custom HTTP client that:
- Mimics the Gemini `generate_content()` interface via a `GroqResponseMock` wrapper
- Supports both batch and streaming completion modes
- Calls Groq's OpenAI-compatible `/v1/chat/completions` endpoint directly
- Uses SSE parsing for streaming responses

---

## 7. Deployment Modes

### 7.1 Standalone Mode (Default)

```
streamlit run app.py
```

The Streamlit app detects that no FastAPI server is running at `localhost:8001` and loads the RAG pipeline directly in-process:

```python
# app.py auto-detection
try:
    health_resp = requests.get(HEALTH_URL, timeout=1.5)
    if health_resp.status_code != 200:
        use_direct_pipeline = True
except Exception:
    use_direct_pipeline = True
```

### 7.2 API + Frontend Mode (Production)

**Terminal 1 — Start the API server:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

**Terminal 2 — Start the frontend:**
```bash
streamlit run app.py
```

The frontend communicates with the API via:
- `POST /chat` — Synchronous request/response
- `POST /chat/stream` — NDJSON streaming response
- `GET /health` — System health check
- `DELETE /session/{id}` — Clear conversation history

---

## 8. Conversation Memory

**Manager:** `src/conversation.py` → `ConversationManager`

```
- In-memory dictionary keyed by session UUID
- Sliding window: keeps last 10 turns (20 messages)
- Used to inject conversation history into the RAG prompt
- Prompt uses last 6 turns (12 messages) for context window efficiency
- Session can be reset via UI button or DELETE /session/{id} API
```

---

## 9. Guardrail Summary — All Defense Layers

```
LAYER 0 ─ INPUT GUARDRAIL (Pre-pipeline)
    ├── L1: Regex signature matching (8 patterns)
    └── L2: LLM adversarial classification (Gemini → Groq)

LAYER 1 ─ PRE-GENERATION GUARDRAIL (Inside response_generator)
    └── Semantic similarity threshold check (score >= 0.40)

LAYER 2 ─ POST-GENERATION GUARDRAIL (output_guardrail_node)
    ├── Price fabrication detection (currency + digits)
    ├── Cost expression detection ("price is ₹X")
    ├── Phone number fabrication (10+ digits not in KB)
    └── URL domain whitelist enforcement

LAYER 3 ─ SYSTEM PROMPT GROUNDING RULES (Embedded in prompt)
    ├── "Answer ONLY using CONTEXT"
    ├── "NEVER invent prices"
    ├── "NEVER provide payment links"
    ├── "No emojis"
    └── "Respond in guest's language"
```

---

## 10. File-to-Component Mapping

| File | Component | Role |
|------|-----------|------|
| `src/graph.py` | LangGraph StateGraph | Defines all 6 nodes, edges, conditional routing, and compiles the workflow |
| `src/rag_pipeline.py` | RAGPipeline | Wrapper class providing `.run()` (batch) and `.run_stream()` (streaming) interfaces |
| `src/retriever.py` | HotelRetriever + SimpleBM25 | Hybrid search: FAISS dense + BM25 sparse + RRF fusion |
| `src/input_guardrail.py` | Input Safety Scanner | Regex signatures + LLM-based adversarial detection |
| `src/guardrail.py` | Output Guardrail | Similarity threshold + hallucination pattern scanning |
| `src/intent_classifier.py` | IntentClassifier | 5-class intent classification via Gemini (Groq fallback) |
| `src/language_detector.py` | Language Detector | Heuristic cascade + Gemini fallback for en/hi/hinglish |
| `src/groq_client.py` | GroqClient | Zero-dependency HTTP client for Groq's OpenAI-compatible API |
| `src/conversation.py` | ConversationManager | In-memory sliding-window session history |
| `src/ingest.py` | Index Builder | Batch-embeds KB → builds FAISS index (run once offline) |
| `src/logger.py` | Logger | Centralized logging configuration |
| `api/main.py` | FastAPI Server | REST API with streaming, health checks, session management |
| `app.py` | Streamlit Frontend | Premium chat UI with telemetry dashboard |
| `data/hotel_kb.json` | Knowledge Base | Structured hotel information corpus |

---

## 11. Data Flow — Complete Request Lifecycle

```
Guest types: "What time does the swimming pool open?"
    │
    ▼
┌─ INPUT GUARDRAIL ─────────────────────────────────────────────────┐
│  L1 Regex: No adversarial patterns matched                       │
│  L2 LLM:   No suspicious keywords → skip LLM scan               │
│  Result:   SAFE → continue                                       │
└──────────────────────────────────────────────────────────────────-┘
    │
    ▼
┌─ LANGUAGE DETECTOR ───────────────────────────────────────────────┐
│  Step 1: No Devanagari characters                                │
│  Step 2: No Hinglish markers                                     │
│  Step 3: Pure ASCII → "en"                                       │
└──────────────────────────────────────────────────────────────────-┘
    │
    ▼
┌─ INTENT CLASSIFIER ──────────────────────────────────────────────┐
│  Gemini prompt: "Classify... swimming pool open..."              │
│  Response: "amenity_question"                                    │
│  Validated: ✓ in VALID_INTENTS                                   │
└──────────────────────────────────────────────────────────────────-┘
    │
    ▼
┌─ HYBRID RETRIEVER ───────────────────────────────────────────────┐
│  FAISS Search:  [Pool Doc (0.72), Gym Doc (0.55), ...]          │
│  BM25 Search:   [Pool Doc (3.1), Schedule Doc (1.8), ...]       │
│  RRF Fusion:    [Pool Doc (0.0328), Schedule Doc (0.0164), ...] │
│  Top-4 returned                                                  │
└──────────────────────────────────────────────────────────────────-┘
    │
    ▼
┌─ RESPONSE GENERATOR ─────────────────────────────────────────────┐
│  Pre-gen guardrail: Top doc score 0.72 >= 0.40 → PASS           │
│  Prompt built with: context + history + intent + language        │
│  Gemini 2.5 Flash generates grounded answer                     │
│  Response: "The swimming pool is open from 6:00 AM to 10:00 PM" │
└──────────────────────────────────────────────────────────────────-┘
    │
    ▼
┌─ OUTPUT GUARDRAIL ───────────────────────────────────────────────┐
│  Price scan:    No currency patterns → PASS                      │
│  Cost scan:     No cost expressions → PASS                       │
│  Phone scan:    No fabricated numbers → PASS                     │
│  URL scan:      No unauthorized URLs → PASS                      │
│  Result:        SAFE → return original response                   │
└──────────────────────────────────────────────────────────────────-┘
    │
    ▼
Guest sees: "The swimming pool is open from 6:00 AM to 10:00 PM..."
```

---

## 12. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **LangGraph over LangChain Chains** | State machine allows conditional routing (e.g., skip pipeline on adversarial input) which is impossible with linear chains |
| **FAISS + BM25 Hybrid** | Semantic search misses exact keyword matches; BM25 misses synonyms. Combining via RRF gets the best of both |
| **Zero-dependency BM25** | Avoids heavy libraries like `rank_bm25`; the custom implementation is ~70 lines and production-ready |
| **Groq as failover (not primary)** | Gemini has better grounding and instruction following; Groq provides ultra-low-latency backup when Gemini is rate-limited |
| **Regex guardrails before LLM guardrails** | Regex is instant and free; LLM classification only triggers on suspicious inputs to minimize API costs |
| **Post-generation hallucination scan** | Even with strong prompts, LLMs can fabricate prices/URLs; regex scanning catches these before the user sees them |
| **Cosine similarity threshold (0.40)** | If no retrieved doc is semantically relevant, generating a response would hallucinate; instead, hand off to human staff |
| **In-memory sessions (not Redis)** | Simplicity for single-instance deployment; easy to swap for Redis/DB in production |
