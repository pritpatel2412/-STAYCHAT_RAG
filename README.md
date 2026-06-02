# Stateful LangGraph Hotel RAG Bot — Grounded AI Concierge

An enterprise-grade, state-of-the-art hotel concierge chatbot built using **LangGraph**, **Python 3.11+**, **Google Gemini 2.5 Flash**, **FAISS Semantic Search**, **BM25 Keyword Search**, **FastAPI**, and **Streamlit**.

This system represents a production-ready Conversational RAG agent. It implements stateful graph workflows, high-precision hybrid document retrieval, adversarial input prompt injection guardrails, and dynamic multi-LLM API failover redundancy.

---

## 🏗️ Stateful LangGraph Node Architecture

The application is structured as a compiled **LangGraph StateGraph** state machine. Each transaction is represented by an acyclic flow executing isolated nodes over a shared transaction state dictionary `AgentState`:

```
                       Guest Message
                             │
                             ▼
                 ┌───────────────────────┐
                 │   [input_guardrail]   │──► Refuses Prompt Injection / Ad-Spam
                 └───────────────────────┘
                             │
                       (Passed Safe)
                             │
                             ▼
                 ┌───────────────────────┐
                 │  [language_detector]  │──► Local ASCII/Devanagari + LLM Fallback
                 └───────────────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  [intent_classifier]  │──► Zero-shot classifier (with Groq Fallback)
                 └───────────────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │       [retriever]     │──► HYBRID: FAISS Semantic + BM25 Keyword
                 └───────────────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  [response_generator] │──► RAG generation (with Groq Fallback)
                 └───────────────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  [output_guardrail]   │──► Anti-hallucination scan & domain checks
                 └───────────────────────┘
                             │
                             ▼
                       Safe Response
```

### Node Explanations
1. **`input_guardrail`:** Scans input for prompt injections (e.g. *"ignore prior guidelines"*) or vulgarity. If flagged, routes directly to a `refusal` response, saving downstream latency and cost.
2. **`language_detector`:** Multi-lingual detector (excluding `"please"` to fix Hinglish bugs) with local ASCII-pacing.
3. **`intent_classifier`:** Routes intent to `booking_inquiry`, `amenity_question`, `complaint`, `staff_command`, or `other` using Gemini (automatically falling back to Groq LLaMA under rate limits).
4. **`retriever`:** Runs **Hybrid Search Retrieval** combining FAISS dense matching and BM25 sparse matching with Reciprocal Rank Fusion (RRF).
5. **`response_generator`:** Generates grounded answers using Gemini (automatically falling back to Groq under rate-limits).
6. **`output_guardrail`:** Executes regex post-gen safety checks (hallucinated pricing, phone numbers, and whitelisted domain URLs).

---

## 🛠️ Advanced RAG Features

### 1. Hybrid Search (FAISS + BM25) with Reciprocal Rank Fusion (RRF)
To provide extremely high keyword and semantic accuracy, the retriever executes a unified hybrid rank fusion:
* **Dense Semantic Match:** Employs cosine-similarity vectors matched against `models/gemini-embedding-2`.
* **Sparse Keyword Match:** Employs a custom python-native **BM25 algorithm** mapping exact terminology (e.g. searching exact digits like `"204"` or codes like `"extension 0"`).
* **Fusion:** Combines ranks mathematically using Reciprocal Rank Fusion (RRF):
  $$RRF\_Score(d) = \frac{1}{60 + r_{semantic}(d)} + \frac{1}{60 + r_{bm25}(d)}$$
  This ranks highly relevant documents at the top, preventing rank inversions.

### 2. Adversarial Input Prompt Guardrails
A dedicated entry node protecting the system against malicious queries:
* Rejects prompt injection attempts (e.g. *"Ignore prior instructions"*, *"System Override"*).
* Blocks off-topic social hacks and vulgarities.
* Instantly returns a polite refusal without invoking costly downstream LLM nodes.

### 3. Multi-LLM API Failover Redundancy (Gemini ──► Groq Fallback)
Free tier limits on Google AI Studio keys (15 requests per minute) can block scaled operations. To guarantee **100% service uptime**, the graph features a built-in rate-limit failover handler:
* If a Gemini call receives a `429 Quota Exceeded` or any network error, the node catches the exception and **instantly routes the request to Groq** using LLaMA (`llama-3.3-70b-versatile` or `llama3-8b-8192`)!
* This provides seamless failover in milliseconds, keeping the Streamlit UI active and completely solving rate limit issues!

---

## 🚀 Setup & Execution

### 1. Environment Setup
```bash
# Clone the repository and activate virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies (updated with LangGraph libraries)
pip install -r requirements.txt
```

### 2. Configure Credentials
Configure your Gemini API key and Groq API key inside **`.env`**:
```bash
# Get Gemini key from Google AI Studio: https://aistudio.google.com/
GEMINI_API_KEY=AIzaSy...

# Get Groq key from Groq Console: https://console.groq.com/
GROQ_API_KEY=gsk_...
```

### 3. Build Hybrid Database Index (Run once)
Compile the vector database files:
```bash
python scripts/build_index.py
```

### 4. Start backend API Server
```bash
uvicorn api.main:app --reload
```
Exposes the backend REST endpoints on `http://localhost:8000`. You can inspect the interactive OpenAPI documentation on `http://localhost:8000/docs`.

### 5. Launch UI Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` to access your premium concierge dashboard!

---

## 🧪 Completed Unit Validation Results

The application compiles and executes **15/15 unit tests** successfully:

```bash
platform win32 -- Python 3.12.10, pytest-8.3.4, pluggy-1.6.0
collected 15 items

tests\test_advanced.py ..                                                [ 13%]
tests\test_guardrail.py .......                                          [ 60%]
tests\test_pipeline.py ....                                              [ 86%]
tests\test_retriever.py ..                                               [100%]

============================= 15 passed in 3.14s ==============================
```

To run the evaluations check locally (paced to stay below free quotas):
```bash
python eval/run_eval.py
```
