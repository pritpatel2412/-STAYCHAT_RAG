# Hotel RAG Bot — Grounded AI Concierge

A production-ready, highly fluent, and grounded hotel concierge chatbot built using **Python 3.11+**, **Google Gemini 1.5 Flash**, **FAISS**, **FastAPI**, and **Streamlit**.

This system strictly grounds its answers in a hotel knowledge base (KB) to prevent hallucinations. It leverages a two-tier defense mechanism to intercept fabricated prices, package costs, phone numbers, or unverified booking links, seamlessly offering a warm multilingual human handoff when required.

---

## 🏗️ Architecture Design & Message Flow

```
   Guest Message
         │
         ▼
 ┌───────────────┐
 │   Language    │──► (EN / HI / Hinglish) Heuristic + Gemini Fallback
 │   Detection   │
 └───────────────┘
         │
         ▼
 ┌───────────────┐
 │    Intent     │──► booking_inquiry | amenity_question | complaint | etc.
 │ Classification│
 └───────────────┘
         │
         ▼
 ┌───────────────┐
 │ FAISS Search  │──► Retrieve Top-4 Context Docs (models/text-embedding-004)
 │  (Retriever)  │
 └───────────────┘
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ [Guardrail Level 1]: Cosine Similarity Check (>= 0.40)  │
 └────────────────────────────────────────────────────────┘
         │                                    │
    (Score >= 0.40)                       (Score < 0.40)
         │                                    │
         ▼                                    ▼
 ┌───────────────┐                  ┌───────────────────┐
 │    Gemini     │                  │   Human Handoff   │──► Final safe response
 │  Generation   │                  │ (Guest Language)  │
 └───────────────┘                  └───────────────────┘
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ [Guardrail Level 2]: Post-Gen Syntactic Regex Scanner  │
 └────────────────────────────────────────────────────────┘
         │                                    │
    (Passed Scan)                       (Fabrication Detected)
         │                                    │
         ▼                                    ▼
 ┌───────────────┐                  ┌───────────────────┐
 │ Return Safe   │                  │ Trigger Hand-off  │──► Final safe response
 │  Response     │                  │   Fallback Block  │
 └───────────────┘                  └───────────────────┘
```

---

## 🛠️ Multi-Tier Anti-Hallucination Guardrails

The bot is designed to **never** fabricate room prices, reservation URLs, or contact numbers. It achieves this with an advanced two-tier safety framework:

1. **Level 1: Semantic Context Check (Pre-Generation)**
   * Every guest query is converted into a vector embedding and run against the local FAISS index.
   * If the similarity score of the top-ranking match falls below **`0.40`**, the query is deemed out-of-bounds (not covered in the KB). Generation is bypassed entirely, returning a polite handoff to human front desk agents in the guest's language.

2. **Level 2: Syntactic Regex Scan (Post-Generation)**
   * If generation occurs, the resulting text undergoes rigorous regex matching before returning to the client:
     * **Price Fabrication Catch:** Employs advanced lookbehinds `(?<=^|\s|[^a-zA-Z0-9])` to accurately capture dollar `$`, Euro `€`, or pound `£` values (fixing standard boundary boundary bugs).
     * **Cost Sentence Catch:** Scans for statements claiming prices (e.g. `costs Rs. 5000`) while allowing generic semantic expressions like *"the price is not listed in our database"* (preventing standard false-positives).
     * **Domain Whitelist Catch:** Intercepts general URLs and checks them against a strict whitelist (`grandhotel.com`, `google.com`, `maps.google`). Any custom domain or payment gateway (e.g. `pay-grandhotel.net`) immediately triggers the fallback.
     * **Phone Numbers Catch:** Identifies any unauthorized 10+ digit blocks, excluding whitelisted concierge lines.

---

## 🗣️ Sophisticated Multilingual Handling

* **Devanagari Check:** Identifies Hindi character ranges to instantly return responses in the native script.
* **Hinglish Keyword check:** Scans for Romanized Hindi keywords (excluding the English word *"please"* to avoid common false positives).
* **LLM Fallback:** If local heuristics are ambiguous, a rapid lightweight call to Gemini validates if the message is in English, Hindi, or Hinglish, responding in the exact tongue of the guest.

---

## 🚀 Setup & Execution

Follow these steps to run the complete environment locally:

### 1. Environment Initialization
Clone the repository and set up a virtual environment:
```bash
# Set up Python virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy the environment template and insert your actual Google Gemini API key:
```bash
cp .env.example .env
# Edit .env and enter GEMINI_API_KEY=AIzaSy...
```

### 3. Build Vector Index (Run once)
Generate vector embeddings and create the FAISS index:
```bash
python scripts/build_index.py
```
This batched ingestion script calls the embedding API in **one optimized call** (up to 20x faster than sequential iterations) and outputs the index in `faiss_index/`.

### 4. Start backend API Server
Fire up the FastAPI app with live reloading:
```bash
uvicorn api.main:app --reload
```
API endpoints are exposed on `http://localhost:8000`. You can inspect the interactive OpenAPI documentation on `http://localhost:8000/docs`.

### 5. Launch User Interface
In a separate terminal, launch the Streamlit chat app:
```bash
streamlit run app.py
```
The app will open automatically on `http://localhost:8501`.

---

## 📈 Evaluation Matrix Summary

The system is validated against 10 distinct testing profiles in `eval/eval_questions.json`:

| Test ID | Query Message | Expected Intent | Tongue | Grounded? | Verification Status |
|---|---|---|---|---|---|
| **eval_01** | *What time does the swimming pool open?* | amenity_question | English | Yes | Answered pool timing (6 AM - 10 PM) |
| **eval_02** | *Mujhe spa book karni hai, kab available hai?* | amenity_question | Hinglish | Yes | Answered spa details in Hinglish |
| **eval_03** | *क्या होटल में पार्किंग उपलब्ध है? कितना शुल्क है?* | amenity_question | Hindi | Yes | Answered parking in Hindi |
| **eval_04** | *What is the price for a Deluxe Room per night?* | booking_inquiry | English | **No (Trap)** | Intercepted! Human handoff returned |
| **eval_05** | *Can you send me a payment link to book a room?* | booking_inquiry | English | **No (Trap)** | Intercepted! Human handoff returned |
| **eval_06** | *My room AC is not working. I want to complain.* | complaint | English | Yes | Routed to Guest Relations Manager |
| **eval_07** | *Please send extra towels to room 412.* | staff_command | English | Yes | Routed to Front Desk (ext 0) |
| **eval_08** | *What is your hotel's cancellation policy?* | booking_inquiry | English | Yes | Answered 48-hour policy |
| **eval_09** | *Do you have a casino or gambling facility?* | amenity_question | English | **No (Trap)** | Intercepted! Returned "Not in KB" |
| **eval_10** | *Gym kitne baje tak khula rehta hai?* | amenity_question | Hinglish | Yes | Answered Gym details in Hinglish |

Run the automated validation report suite to view these checks locally:
```bash
python eval/run_eval.py
```

---

## 💡 Key Design Decisions & Production Upgrades
* **Batched Ingestion:** In `src/ingest.py`, passing a complete text list to the embeddings API reduces latency and rate-limiting limits.
* **Component Lazy-Loading:** The `HotelRetriever` handles server starts when index files are missing, recovering automatically once the ingestion script is executed.
* **In-Memory History Memory:** Built with sliding-window trims. In scaled production environments, replace `ConversationManager` with a distributed cache like **Redis**.
* **Zero-Dependency Cosine Matching:** Normalized inner-product index `faiss.IndexFlatIP` yields mathematically exact cosine similarity without heavy cloud server weights.
