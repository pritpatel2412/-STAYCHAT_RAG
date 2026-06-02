# Hotel RAG Bot — Full Build Prompt

> Paste this entire document as your prompt to the coding agent (Cursor, Bolt, Lovable, Replit Agent, etc.)

---

## Project Overview

Build a production-ready, grounded hotel concierge chatbot using Python, Google Gemini, and FAISS. The bot answers guest questions strictly from a hotel knowledge base (KB). It must never invent prices, policies, or payment links. If information is not in the KB, it must say so and offer to connect the guest to a human agent.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM | Google Gemini 1.5 Flash (via `google-generativeai` SDK) |
| Embeddings | `models/text-embedding-004` (Google) |
| Vector Store | FAISS (`faiss-cpu`) |
| API Framework | FastAPI + Uvicorn |
| Frontend (optional) | Streamlit (simple chat UI) |
| Env Management | `python-dotenv` |
| Testing | `pytest` |

---

## Project Structure

```
hotel-rag-bot/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── hotel_kb.json          # Hotel knowledge base
├── src/
│   ├── __init__.py
│   ├── ingest.py              # KB ingestion + FAISS index builder
│   ├── retriever.py           # FAISS vector search
│   ├── intent_classifier.py   # Intent classification (Gemini)
│   ├── guardrail.py           # Anti-hallucination guardrail
│   ├── rag_pipeline.py        # Main RAG pipeline (retrieval + generation)
│   ├── conversation.py        # Multi-turn conversation manager
│   └── language_detector.py  # Language detection (EN/HI/Hinglish)
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI app
├── app.py                     # Streamlit chat UI
├── eval/
│   ├── eval_questions.json    # 10 eval questions (incl. trap questions)
│   └── run_eval.py            # Eval runner
├── tests/
│   ├── test_retriever.py
│   ├── test_guardrail.py
│   └── test_pipeline.py
└── scripts/
    └── build_index.py         # One-time index build script
```

---

## Step 1 — Hotel Knowledge Base (`data/hotel_kb.json`)

Create a realistic, comprehensive KB with at least 30 entries covering the following categories. Each entry must have `id`, `category`, `title`, and `content` fields.

```json
[
  {
    "id": "amenity_001",
    "category": "amenity",
    "title": "Swimming Pool",
    "content": "The outdoor swimming pool is open daily from 6:00 AM to 10:00 PM. The pool area includes sun loungers, towel service, and a poolside bar serving light snacks and beverages. Children under 12 must be accompanied by an adult. Pool depth ranges from 1.2m to 2.4m."
  },
  {
    "id": "amenity_002",
    "category": "amenity",
    "title": "Spa and Wellness Center",
    "content": "The Serenity Spa is open Monday to Sunday, 9:00 AM to 9:00 PM. Services include Swedish massage, deep tissue massage, aromatherapy, facial treatments, and a steam room. Advance booking is required and can be done at the front desk or by calling extension 205. The spa does not accept walk-in appointments on weekends."
  },
  {
    "id": "amenity_003",
    "category": "amenity",
    "title": "Fitness Center",
    "content": "The 24-hour fitness center is located on the 2nd floor. Equipment includes treadmills, ellipticals, free weights, resistance machines, and yoga mats. A certified personal trainer is available on weekday mornings from 7:00 AM to 11:00 AM. No booking required for gym access; use your room key card."
  },
  {
    "id": "dining_001",
    "category": "dining",
    "title": "The Grand Restaurant — Hours and Overview",
    "content": "The Grand Restaurant on the lobby level serves breakfast (7:00 AM – 10:30 AM), lunch (12:30 PM – 3:00 PM), and dinner (7:00 PM – 11:00 PM). The menu features Indian, Continental, and Asian cuisine. The restaurant accommodates both in-house guests and outside visitors. Dress code: smart casual for dinner."
  },
  {
    "id": "dining_002",
    "category": "dining",
    "title": "Room Service",
    "content": "Room service is available 24 hours a day, 7 days a week. To place an order, dial extension 204 from your room phone or use the in-room tablet. Standard delivery time is 30–45 minutes. A room service surcharge of INR 150 applies per order. The full room service menu is available in your room's welcome folder."
  },
  {
    "id": "dining_003",
    "category": "dining",
    "title": "Rooftop Bar — Skyline Lounge",
    "content": "Skyline Lounge on the 14th floor is open Wednesday to Sunday from 5:00 PM to 12:00 AM. The lounge serves cocktails, mocktails, wines, and a curated selection of bar snacks. Live acoustic music plays every Friday and Saturday from 7:30 PM. Minimum age: 21 years. Smart casual dress code enforced."
  },
  {
    "id": "checkin_001",
    "category": "check_in_out",
    "title": "Check-in and Check-out Times",
    "content": "Standard check-in time is 2:00 PM. Standard check-out time is 12:00 noon. Early check-in (from 10:00 AM) is subject to availability and may incur an additional half-day charge. Late check-out (until 6:00 PM) is available upon request and subject to availability; charges may apply. Please contact the front desk for assistance."
  },
  {
    "id": "checkin_002",
    "category": "check_in_out",
    "title": "Express Check-out",
    "content": "Express check-out is available for guests who have provided a credit card at the time of check-in. Your itemized bill will be emailed to the address provided during booking. Simply leave your room key at the front desk or drop it in the express check-out box located in the lobby before noon."
  },
  {
    "id": "rooms_001",
    "category": "rooms",
    "title": "Room Types Overview",
    "content": "The hotel offers four room categories: Deluxe Room (single or double occupancy, garden or pool view), Club Room (upper floors, complimentary Club Lounge access, city view), Junior Suite (separate living area, premium amenities), and Presidential Suite (full floor, private butler service, panoramic views). All rooms include high-speed Wi-Fi, air conditioning, flat-screen TV, minibar, and in-room safe."
  },
  {
    "id": "rooms_002",
    "category": "rooms",
    "title": "Wi-Fi Access",
    "content": "Complimentary high-speed Wi-Fi is available throughout the hotel. Network name: HotelGuest_5G. Password is available at the front desk or on your room key card envelope. Wi-Fi speed is up to 100 Mbps. For technical assistance, call IT support at extension 209."
  },
  {
    "id": "rooms_003",
    "category": "rooms",
    "title": "Minibar Policy",
    "content": "All rooms come with a complimentary stocked minibar including water, soft drinks, and two alcoholic beverages. Consumed items are automatically billed to your room account via RFID sensors. Minibar is restocked daily. Additional items can be requested from room service."
  },
  {
    "id": "policy_001",
    "category": "policy",
    "title": "Cancellation Policy",
    "content": "Reservations cancelled more than 48 hours before the check-in date will receive a full refund to the original payment method. Cancellations within 48 hours of check-in will be charged one night's stay as a cancellation fee. No-show bookings are charged the full reservation amount. To cancel, contact the reservations team at reservations@grandhotel.com or call the front desk."
  },
  {
    "id": "policy_002",
    "category": "policy",
    "title": "Pet Policy",
    "content": "The hotel is pet-friendly. Well-behaved dogs and cats are welcome in designated pet-friendly rooms (subject to availability) for an additional fee of INR 1,500 per night. Pets must be kept on a leash in public areas. Pets are not permitted in the restaurant, pool area, or spa. Please inform the reservations team when booking if you are bringing a pet."
  },
  {
    "id": "policy_003",
    "category": "policy",
    "title": "Smoking Policy",
    "content": "The hotel is entirely non-smoking indoors, including all guest rooms, corridors, restaurants, and public areas. Designated smoking zones are located near the main entrance and at the poolside garden area. A deep-cleaning fee of INR 5,000 will be charged for smoking in non-designated areas."
  },
  {
    "id": "services_001",
    "category": "services",
    "title": "Concierge Services",
    "content": "Our concierge team is available at the lobby desk 24 hours a day. Services include restaurant reservations, sightseeing tours, car rental arrangements, airport transfers, laundry and dry-cleaning coordination, and courier services. The concierge can also assist with sourcing local products and arranging customized experiences."
  },
  {
    "id": "services_002",
    "category": "services",
    "title": "Laundry and Dry Cleaning",
    "content": "Laundry service is available daily. Items left before 9:00 AM will be returned by 6:00 PM the same day (same-day service). Items left after 9:00 AM will be returned the next day. Express service (4-hour turnaround) is available for an additional charge. Place items in the laundry bag in your wardrobe and fill out the laundry slip."
  },
  {
    "id": "services_003",
    "category": "services",
    "title": "Airport Transfer",
    "content": "The hotel offers airport transfer services to and from Chhatrapati Shivaji Maharaj International Airport. Transfers can be booked through the concierge desk or at check-in. Standard sedan transfer is available; luxury vehicle upgrades are also available. Please provide your flight details at least 6 hours in advance. Contact the concierge at extension 201."
  },
  {
    "id": "facilities_001",
    "category": "facilities",
    "title": "Business Center",
    "content": "The Business Center is open Monday to Friday from 8:00 AM to 8:00 PM, and Saturday from 9:00 AM to 3:00 PM. It is closed on Sundays. Facilities include workstations, high-speed internet, printing, scanning, photocopying, and a private meeting pod for up to 4 people. Located on the mezzanine floor."
  },
  {
    "id": "facilities_002",
    "category": "facilities",
    "title": "Meeting and Event Spaces",
    "content": "The hotel has three conference rooms: Sapphire Hall (capacity 80 pax, boardroom or classroom layout), Emerald Room (capacity 30 pax, ideal for workshops), and the Executive Boardroom (capacity 12 pax, permanent boardroom setup). All rooms include AV equipment, whiteboards, and catering coordination. For event bookings, contact the events team at events@grandhotel.com."
  },
  {
    "id": "facilities_003",
    "category": "facilities",
    "title": "Parking",
    "content": "The hotel provides complimentary valet parking for all in-house guests. Self-parking is available in the basement at INR 100 per hour or INR 600 per day. Valet parking requests should be submitted at the front desk at least 30 minutes before vehicle retrieval. Electric vehicle charging stations are available on B2 level."
  },
  {
    "id": "location_001",
    "category": "location",
    "title": "Hotel Location and Directions",
    "content": "The Grand Hotel is located at 42 Marine Drive Boulevard, South Mumbai, Maharashtra 400002. We are a 5-minute walk from Marine Lines railway station. The nearest metro station is Chhatrapati Shivaji Terminus (CST) on the Aqua Line, a 10-minute walk. From the airport, the hotel is approximately 45 minutes by car under normal traffic conditions."
  },
  {
    "id": "location_002",
    "category": "location",
    "title": "Nearby Attractions",
    "content": "Within walking distance: Gateway of India (15 min), Taj Mahal Palace (12 min), Chhatrapati Shivaji Maharaj Vastu Sangrahalaya museum (10 min walk). Nearby by car: Juhu Beach (25 min), Elephanta Caves ferry terminal (20 min drive). The concierge can arrange guided tours to any of these destinations."
  },
  {
    "id": "accessibility_001",
    "category": "accessibility",
    "title": "Accessibility Features",
    "content": "The hotel is fully accessible for guests with disabilities. All public areas, restaurants, and the pool are wheelchair accessible. Accessible guest rooms with widened doorways, roll-in showers, grab bars, and lowered amenities are available on request. Please inform the reservations team of specific accessibility requirements at the time of booking."
  },
  {
    "id": "safety_001",
    "category": "safety",
    "title": "Safety and Emergency Procedures",
    "content": "In case of emergency, dial extension 0 from any hotel phone or call the front desk at +91-22-6600-0000. Emergency exits are clearly marked on each floor. Fire evacuation procedures are posted on the back of every room door. The hotel has 24-hour security. A defibrillator is located at the front desk. First aid is available at the concierge desk."
  },
  {
    "id": "children_001",
    "category": "family",
    "title": "Children's Facilities and Babysitting",
    "content": "The hotel has a dedicated Kids' Club on the ground floor, open daily from 10:00 AM to 6:00 PM for children aged 4 to 12. Activities include arts and crafts, games, and supervised play. Babysitting services can be arranged through the concierge with 24 hours' advance notice. An extra bed or cot for children under 5 is provided free of charge."
  },
  {
    "id": "loyalty_001",
    "category": "loyalty",
    "title": "Loyalty Program",
    "content": "The Grand Rewards loyalty program offers three tiers: Silver (0–4 nights/year), Gold (5–14 nights/year), and Platinum (15+ nights/year). Members earn 10 points per INR 100 spent and can redeem points for room upgrades, dining credits, and spa treatments. Enroll at the front desk or at grandhotel.com/rewards. Points expire after 24 months of inactivity."
  },
  {
    "id": "transport_001",
    "category": "transport",
    "title": "Local Transport Options",
    "content": "The concierge can arrange pre-booked taxis and luxury car services. The hotel does not have a direct affiliation with app-based ride services, but guests may use any ride-hailing app. Bicycle rentals are available at the concierge desk for INR 200 per hour. A hop-on hop-off tourist bus stop is located 3 minutes from the main entrance."
  },
  {
    "id": "feedback_001",
    "category": "feedback",
    "title": "Guest Feedback and Complaints",
    "content": "We take all guest feedback seriously. For immediate concerns, please contact the duty manager at extension 200, available 24 hours. For written feedback, speak to any team member or complete the feedback form available at the front desk or in your room. Our Guest Relations Manager, Ms. Priya Sharma, is available weekdays from 9:00 AM to 6:00 PM."
  },
  {
    "id": "payment_001",
    "category": "payment",
    "title": "Accepted Payment Methods",
    "content": "The hotel accepts Visa, Mastercard, American Express, and RuPay credit and debit cards. Cash payment in Indian Rupees is accepted at the front desk. UPI payments are accepted for incidental charges settled at checkout. Foreign currency exchange is available at the front desk during business hours."
  },
  {
    "id": "dining_004",
    "category": "dining",
    "title": "Dietary Accommodations",
    "content": "The Grand Restaurant offers vegetarian, vegan, gluten-free, Jain, and Halal meal options. Please inform the restaurant host or room service when placing your order. Guests with severe allergies should speak with the head chef directly; contact the restaurant at extension 203."
  }
]
```

---

## Step 2 — FAISS Ingestion (`src/ingest.py`)

```python
"""
Ingestion pipeline: loads hotel KB, generates embeddings via Google Gemini,
and builds a FAISS index. Run once before starting the server.
"""

import json
import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "models/text-embedding-004"
INDEX_PATH = "faiss_index/hotel.index"
META_PATH  = "faiss_index/hotel_meta.pkl"
KB_PATH    = "data/hotel_kb.json"


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using Gemini embeddings. Returns (N, D) float32 array."""
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result["embedding"])
    return np.array(embeddings, dtype="float32")


def build_index():
    os.makedirs("faiss_index", exist_ok=True)

    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    # Build rich text representations for embedding
    texts = [
        f"Category: {item['category']}\nTitle: {item['title']}\n{item['content']}"
        for item in kb
    ]

    print(f"Embedding {len(texts)} KB entries...")
    embeddings = embed_texts(texts)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build flat inner-product index (equivalent to cosine after normalization)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(kb, f)

    print(f"FAISS index built with {index.ntotal} vectors. Saved to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
```

---

## Step 3 — Retriever (`src/retriever.py`)

```python
"""
FAISS-based semantic retriever. ALWAYS uses real vector search —
never passes the full KB into the prompt.
"""

import pickle
import numpy as np
import faiss
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "models/text-embedding-004"
INDEX_PATH = "faiss_index/hotel.index"
META_PATH  = "faiss_index/hotel_meta.pkl"


class HotelRetriever:
    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.kb = pickle.load(f)

    def _embed_query(self, query: str) -> np.ndarray:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        vec = np.array([result["embedding"]], dtype="float32")
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, query: str) -> list[dict]:
        """
        Returns top-K KB entries most relevant to the query.
        Includes similarity score for downstream confidence checks.
        """
        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, self.top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.kb[idx])
            entry["similarity_score"] = float(score)
            results.append(entry)

        return results
```

---

## Step 4 — Intent Classifier (`src/intent_classifier.py`)

```python
"""
Classifies the guest's message into one of five intents using Gemini.
Intents: booking_inquiry | amenity_question | complaint | staff_command | other
"""

import os
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

VALID_INTENTS = {
    "booking_inquiry",
    "amenity_question",
    "complaint",
    "staff_command",
    "other"
}

CLASSIFICATION_PROMPT = """You are a hotel concierge intent classifier.

Classify the guest message into EXACTLY ONE of these intents:
- booking_inquiry: Questions about reservations, check-in, check-out, cancellation, room availability, prices, payment
- amenity_question: Questions about hotel facilities, services, dining, spa, pool, gym, Wi-Fi, parking, etc.
- complaint: Expressing dissatisfaction, reporting a problem, complaining about something
- staff_command: Requesting an action from hotel staff (e.g., "please send towels", "I need housekeeping")
- other: Anything that doesn't fit the above categories

Reply with ONLY the intent label, nothing else.

Guest message: {message}"""


def classify_intent(message: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = CLASSIFICATION_PROMPT.format(message=message)
    response = model.generate_content(prompt)
    intent = response.text.strip().lower()

    # Sanitize — fall back to 'other' if model goes off-script
    if intent not in VALID_INTENTS:
        return "other"
    return intent
```

---

## Step 5 — Anti-Hallucination Guardrail (`src/guardrail.py`)

**This is the most critical component. Read carefully.**

```python
"""
Anti-hallucination guardrail. This module ensures the bot NEVER invents:
  - Room prices or package costs
  - Payment links or booking URLs
  - Contact details not in the KB
  - Policies not in the KB

The guardrail operates at two levels:
  1. Pre-generation: Low similarity score → skip generation, return "not in KB" response
  2. Post-generation: Scan generated text for hallucination patterns
"""

import re

# ─── Thresholds ──────────────────────────────────────────────────────────────
MIN_SIMILARITY_SCORE = 0.40   # Below this → context is too weak → do not generate

# ─── Patterns that should NEVER appear in bot responses ──────────────────────
HALLUCINATION_PATTERNS = [
    # Invented prices (any currency + digits combo NOT preceded by "INR" from KB)
    r"(?i)\b(USD|EUR|GBP|\$|€|£)\s*[\d,]+",
    # Invented payment/booking links
    r"https?://(?!grandhotel\.com|google\.com|maps\.google)",
    # "Book now at [URL]" patterns
    r"(?i)book\s+now\s+at\s+https?://",
    # Invented phone numbers (10+ digit strings not from KB)
    r"\b(?<!\+91-22-6600-)(\d{10,})\b",
    # "Our price is..." fabrication patterns
    r"(?i)(the price is|costs? (?:INR|Rs\.?|₹)\s*[\d,]+(?!\s*per\s*(night|hour|day|order)))",
]

HUMAN_HANDOFF_EN = (
    "I'm sorry, I don't have that information in our hotel knowledge base. "
    "I'd be happy to connect you with a member of our team who can assist you directly. "
    "Please call the front desk at extension 0, or I can have someone reach out to you."
)

HUMAN_HANDOFF_HI = (
    "मुझे खेद है, यह जानकारी हमारे होटल के ज्ञान आधार में उपलब्ध नहीं है। "
    "मैं आपको हमारी टीम के किसी सदस्य से जोड़ सकता हूं जो आपकी सहायता कर सकते हैं। "
    "कृपया एक्सटेंशन 0 पर फ्रंट डेस्क को कॉल करें।"
)

HUMAN_HANDOFF_HINGLISH = (
    "Sorry, yeh information hamare hotel ke knowledge base mein available nahi hai. "
    "Main aapko humare team ke kisi member se connect kar sakta hoon jo aapki help kar sake. "
    "Please front desk ko extension 0 par call karein."
)


def check_similarity_threshold(retrieved_docs: list[dict]) -> bool:
    """Returns True if at least one retrieved doc clears the confidence threshold."""
    if not retrieved_docs:
        return False
    return retrieved_docs[0]["similarity_score"] >= MIN_SIMILARITY_SCORE


def scan_for_hallucinations(text: str) -> list[str]:
    """Returns list of detected hallucination pattern matches."""
    violations = []
    for pattern in HALLUCINATION_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            violations.append(f"Pattern '{pattern}' matched: {matches}")
    return violations


def get_human_handoff(language: str) -> str:
    """Returns the 'not in KB' response in the appropriate language."""
    if language == "hi":
        return HUMAN_HANDOFF_HI
    elif language == "hinglish":
        return HUMAN_HANDOFF_HINGLISH
    else:
        return HUMAN_HANDOFF_EN


def apply_guardrail(
    generated_text: str,
    retrieved_docs: list[dict],
    language: str = "en"
) -> dict:
    """
    Main guardrail function. Returns:
      {
        "safe": bool,
        "text": str,            # final response to return to user
        "reason": str | None    # why it was blocked (for logging)
      }
    """
    # Level 1: Similarity check
    if not check_similarity_threshold(retrieved_docs):
        return {
            "safe": False,
            "text": get_human_handoff(language),
            "reason": f"Low similarity score: {retrieved_docs[0]['similarity_score'] if retrieved_docs else 'no docs'}"
        }

    # Level 2: Post-generation hallucination scan
    violations = scan_for_hallucinations(generated_text)
    if violations:
        return {
            "safe": False,
            "text": get_human_handoff(language),
            "reason": f"Hallucination detected: {violations}"
        }

    return {
        "safe": True,
        "text": generated_text,
        "reason": None
    }
```

---

## Step 6 — Language Detector (`src/language_detector.py`)

```python
"""
Detects language of guest message: 'en', 'hi', or 'hinglish'.
Uses a simple heuristic + Gemini fallback for ambiguous cases.
"""

import re
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Unicode range for Devanagari script (Hindi)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Common Hinglish marker words
HINGLISH_MARKERS = {
    "kya", "hai", "mujhe", "aapka", "kab", "kaise", "hain",
    "chahiye", "batao", "please", "bata", "yahan", "wahan",
    "kitne", "kitna", "accha", "theek"
}


def detect_language(text: str) -> str:
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))

    # Contains Devanagari → Hindi
    if DEVANAGARI_RE.search(text):
        return "hi"

    # Contains Hinglish markers mixed with English
    if words & HINGLISH_MARKERS:
        return "hinglish"

    # Default
    return "en"
```

---

## Step 7 — RAG Pipeline (`src/rag_pipeline.py`)

```python
"""
Main RAG pipeline combining retrieval, intent classification, language detection,
generation, and guardrail enforcement.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

from src.retriever import HotelRetriever
from src.intent_classifier import classify_intent
from src.language_detector import detect_language
from src.guardrail import apply_guardrail, get_human_handoff

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GENERATION_MODEL = "gemini-1.5-flash"

SYSTEM_PROMPT = """You are a professional hotel concierge assistant for The Grand Hotel, Mumbai.

CRITICAL RULES — you MUST follow these without exception:
1. Answer ONLY using the information provided in the CONTEXT section below.
2. NEVER invent, assume, or estimate prices, room rates, or fees that are not explicitly stated in the CONTEXT.
3. NEVER provide payment links, booking URLs, or reservation systems not mentioned in the CONTEXT.
4. If the CONTEXT does not contain enough information to answer the question, say so clearly and offer to connect the guest with a human team member.
5. Do not make up contact details, phone numbers, or email addresses not present in the CONTEXT.
6. Be warm, professional, and concise.
7. Respond in the SAME LANGUAGE as the guest's message ({language_instruction}).

CONTEXT (retrieved from hotel knowledge base):
{context}

CONVERSATION HISTORY:
{history}

INTENT: {intent}
GUEST MESSAGE: {user_message}

Respond to the guest now:"""


def format_context(retrieved_docs: list[dict]) -> str:
    if not retrieved_docs:
        return "No relevant information found."
    sections = []
    for doc in retrieved_docs:
        sections.append(
            f"[{doc['category'].upper()}] {doc['title']}\n{doc['content']}"
        )
    return "\n\n---\n\n".join(sections)


def format_history(history: list[dict]) -> str:
    if not history:
        return "No previous messages."
    lines = []
    for turn in history[-6:]:  # Keep last 3 turns (6 messages)
        role = "Guest" if turn["role"] == "user" else "Concierge"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def get_language_instruction(language: str) -> str:
    return {
        "en": "English",
        "hi": "Hindi (Devanagari script)",
        "hinglish": "Hinglish (Roman script Hindi-English mix)"
    }.get(language, "English")


class RAGPipeline:
    def __init__(self):
        self.retriever = HotelRetriever(top_k=4)
        self.model = genai.GenerativeModel(GENERATION_MODEL)

    def run(
        self,
        user_message: str,
        conversation_history: list[dict]
    ) -> dict:
        """
        Full RAG pipeline.
        Returns:
          {
            "response": str,
            "intent": str,
            "language": str,
            "retrieved_docs": list,
            "guardrail_triggered": bool,
            "guardrail_reason": str | None
          }
        """
        # 1. Detect language
        language = detect_language(user_message)

        # 2. Classify intent
        intent = classify_intent(user_message)

        # 3. Retrieve relevant KB entries via FAISS
        retrieved_docs = self.retriever.retrieve(user_message)

        # 4. Pre-generation guardrail: check similarity threshold
        from src.guardrail import check_similarity_threshold
        if not check_similarity_threshold(retrieved_docs):
            return {
                "response": get_human_handoff(language),
                "intent": intent,
                "language": language,
                "retrieved_docs": retrieved_docs,
                "guardrail_triggered": True,
                "guardrail_reason": "Low similarity — topic not in KB"
            }

        # 5. Build prompt
        context = format_context(retrieved_docs)
        history = format_history(conversation_history)
        lang_instruction = get_language_instruction(language)

        prompt = SYSTEM_PROMPT.format(
            language_instruction=lang_instruction,
            context=context,
            history=history,
            intent=intent,
            user_message=user_message
        )

        # 6. Generate response
        response = self.model.generate_content(prompt)
        generated_text = response.text.strip()

        # 7. Post-generation guardrail: scan for hallucinations
        guardrail_result = apply_guardrail(generated_text, retrieved_docs, language)

        return {
            "response": guardrail_result["text"],
            "intent": intent,
            "language": language,
            "retrieved_docs": [
                {"id": d["id"], "title": d["title"], "score": d["similarity_score"]}
                for d in retrieved_docs
            ],
            "guardrail_triggered": not guardrail_result["safe"],
            "guardrail_reason": guardrail_result["reason"]
        }
```

---

## Step 8 — Conversation Manager (`src/conversation.py`)

```python
"""
Manages per-session conversation history for multi-turn context.
"""

from collections import defaultdict


class ConversationManager:
    def __init__(self, max_turns: int = 10):
        self.sessions: dict[str, list[dict]] = defaultdict(list)
        self.max_turns = max_turns

    def add_turn(self, session_id: str, role: str, content: str):
        self.sessions[session_id].append({"role": role, "content": content})
        # Trim to max_turns (each turn = 2 messages)
        if len(self.sessions[session_id]) > self.max_turns * 2:
            self.sessions[session_id] = self.sessions[session_id][-(self.max_turns * 2):]

    def get_history(self, session_id: str) -> list[dict]:
        return self.sessions.get(session_id, [])

    def clear_session(self, session_id: str):
        self.sessions.pop(session_id, None)
```

---

## Step 9 — FastAPI Application (`api/main.py`)

```python
"""FastAPI REST API for the hotel RAG bot."""

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_pipeline import RAGPipeline
from src.conversation import ConversationManager

app = FastAPI(
    title="Hotel RAG Bot API",
    description="Grounded hotel concierge chatbot powered by Gemini + FAISS",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()
conv_manager = ConversationManager()


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    language: str
    guardrail_triggered: bool
    retrieved_docs: list[dict]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    history = conv_manager.get_history(session_id)
    result = pipeline.run(request.message, history)

    conv_manager.add_turn(session_id, "user", request.message)
    conv_manager.add_turn(session_id, "assistant", result["response"])

    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        intent=result["intent"],
        language=result["language"],
        guardrail_triggered=result["guardrail_triggered"],
        retrieved_docs=result["retrieved_docs"]
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    conv_manager.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

## Step 10 — Streamlit Chat UI (`app.py`)

```python
"""Streamlit frontend for the hotel RAG bot."""

import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="The Grand Hotel — Concierge", page_icon="🏨")
st.title("🏨 The Grand Hotel Concierge")
st.caption("Powered by Gemini + FAISS | Multilingual: English, Hindi, Hinglish")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("🔍 Debug info"):
                st.json(msg["meta"])

if prompt := st.chat_input("Ask me anything about the hotel..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(API_URL, json={
                    "session_id": st.session_state.session_id,
                    "message": prompt
                }, timeout=30)
                data = resp.json()

                st.markdown(data["response"])

                # Show intent + language badge
                col1, col2, col3 = st.columns(3)
                col1.metric("Intent", data["intent"])
                col2.metric("Language", data["language"])
                col3.metric("Guardrail", "🚫 ON" if data["guardrail_triggered"] else "✅ OFF")

                meta = {
                    "intent": data["intent"],
                    "language": data["language"],
                    "guardrail_triggered": data["guardrail_triggered"],
                    "retrieved_docs": data["retrieved_docs"]
                }
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["response"],
                    "meta": meta
                })

            except Exception as e:
                st.error(f"API error: {e}")
```

---

## Step 11 — Eval Set (`eval/eval_questions.json`)

```json
[
  {
    "id": "eval_01",
    "question": "What time does the swimming pool open?",
    "expected_intent": "amenity_question",
    "expected_language": "en",
    "should_answer": true,
    "notes": "Direct KB hit — amenity_001"
  },
  {
    "id": "eval_02",
    "question": "Mujhe spa book karni hai, kab available hai?",
    "expected_intent": "amenity_question",
    "expected_language": "hinglish",
    "should_answer": true,
    "notes": "Hinglish — spa booking info in amenity_002"
  },
  {
    "id": "eval_03",
    "question": "क्या होटल में पार्किंग उपलब्ध है? कितना शुल्क है?",
    "expected_intent": "amenity_question",
    "expected_language": "hi",
    "should_answer": true,
    "notes": "Hindi — parking in facilities_003, price in KB"
  },
  {
    "id": "eval_04",
    "question": "What is the price for a Deluxe Room per night?",
    "expected_intent": "booking_inquiry",
    "expected_language": "en",
    "should_answer": false,
    "notes": "TRAP — room prices are NOT in the KB. Must NOT invent a price."
  },
  {
    "id": "eval_05",
    "question": "Can you send me a payment link to book a room?",
    "expected_intent": "booking_inquiry",
    "expected_language": "en",
    "should_answer": false,
    "notes": "TRAP — no payment links in KB. Must never fabricate a URL."
  },
  {
    "id": "eval_06",
    "question": "My room AC is not working. I want to complain.",
    "expected_intent": "complaint",
    "expected_language": "en",
    "should_answer": true,
    "notes": "Complaint intent — route to feedback/duty manager (feedback_001, safety_001)"
  },
  {
    "id": "eval_07",
    "question": "Please send extra towels to room 412.",
    "expected_intent": "staff_command",
    "expected_language": "en",
    "should_answer": true,
    "notes": "Staff command — should acknowledge and direct to housekeeping (extension 0)"
  },
  {
    "id": "eval_08",
    "question": "What is your hotel's cancellation policy?",
    "expected_intent": "booking_inquiry",
    "expected_language": "en",
    "should_answer": true,
    "notes": "Direct KB hit — policy_001"
  },
  {
    "id": "eval_09",
    "question": "Do you have a casino or gambling facility?",
    "expected_intent": "amenity_question",
    "expected_language": "en",
    "should_answer": false,
    "notes": "TRAP — not in KB. Should say not available and offer human."
  },
  {
    "id": "eval_10",
    "question": "Gym kitne baje tak khula rehta hai aur kya personal trainer milega?",
    "expected_intent": "amenity_question",
    "expected_language": "hinglish",
    "should_answer": true,
    "notes": "Hinglish — fitness center in amenity_003, trainer info present"
  }
]
```

---

## Step 12 — Eval Runner (`eval/run_eval.py`)

```python
"""
Runs the evaluation set and prints a structured report.
Checks: correct intent, correct language detection, guardrail behavior, no hallucination.
"""

import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import RAGPipeline
from src.guardrail import scan_for_hallucinations

pipeline = RAGPipeline()
results = []

with open("eval/eval_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

print("\n" + "="*70)
print("HOTEL RAG BOT — EVALUATION REPORT")
print("="*70)

passed = 0
failed = 0

for q in questions:
    result = pipeline.run(q["question"], [])

    intent_ok     = result["intent"] == q["expected_intent"]
    language_ok   = result["language"] == q["expected_language"]
    hallucination = scan_for_hallucinations(result["response"])

    # For trap questions: guardrail MUST trigger OR response must say "not available"
    if not q["should_answer"]:
        guardrail_ok = (
            result["guardrail_triggered"] or
            any(phrase in result["response"].lower() for phrase in [
                "not in", "don't have", "nahi hai", "unavailable",
                "connect you", "human", "front desk", "team member"
            ])
        )
    else:
        guardrail_ok = True  # For normal questions, guardrail should NOT trigger

    no_hallucination = len(hallucination) == 0
    overall_pass = intent_ok and language_ok and guardrail_ok and no_hallucination

    if overall_pass:
        passed += 1
        status = "✅ PASS"
    else:
        failed += 1
        status = "❌ FAIL"

    print(f"\n[{q['id']}] {status}")
    print(f"  Question    : {q['question'][:80]}")
    print(f"  Intent      : {result['intent']} (expected: {q['expected_intent']}) {'✓' if intent_ok else '✗'}")
    print(f"  Language    : {result['language']} (expected: {q['expected_language']}) {'✓' if language_ok else '✗'}")
    print(f"  Guardrail   : triggered={result['guardrail_triggered']} | ok={guardrail_ok} {'✓' if guardrail_ok else '✗'}")
    print(f"  Hallucinate : {hallucination if hallucination else 'none'} {'✓' if no_hallucination else '✗'}")
    print(f"  Response    : {result['response'][:120]}...")

print("\n" + "="*70)
print(f"RESULT: {passed}/{len(questions)} passed | {failed}/{len(questions)} failed")
print("="*70 + "\n")
```

---

## Step 13 — requirements.txt

```
google-generativeai>=0.7.0
faiss-cpu>=1.7.4
numpy>=1.26.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
streamlit>=1.35.0
python-dotenv>=1.0.0
pydantic>=2.7.0
pytest>=8.2.0
httpx>=0.27.0
requests>=2.32.0
```

---

## Step 14 — .env.example

```
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Step 15 — README.md

Write a complete README with the following sections:

1. **Project Overview** — What the bot does, the key features (grounded RAG, anti-hallucination, multilingual, intent classification), and the tech stack table.

2. **Architecture Diagram (text)** — ASCII or Mermaid diagram showing:
   ```
   Guest Message
       ↓
   Language Detection → Intent Classification
       ↓
   FAISS Retriever (real vector search, top-4)
       ↓
   Similarity Threshold Check (guardrail level 1)
       ↓
   Gemini Generation (context-grounded prompt)
       ↓
   Hallucination Scanner (guardrail level 2)
       ↓
   Response → Guest
   ```

3. **Anti-Hallucination Guardrail** — Explain the two-level approach clearly:
   - Level 1: Cosine similarity threshold (0.40). If no retrieved doc meets this, skip generation entirely and return human handoff.
   - Level 2: Post-generation regex scan for invented prices, foreign currency, non-whitelisted URLs, fabricated phone numbers.
   - Why this matters: the bot must NEVER invent a room price or payment link.

4. **Intent Classification** — Five intents, examples of each.

5. **Multilingual Support** — How EN/HI/Hinglish is detected and how the response language matches the input.

6. **Setup Instructions**:
   ```bash
   git clone <repo>
   cd hotel-rag-bot
   python -m venv venv
   source venv/bin/activate       # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   # Add GEMINI_API_KEY to .env
   python scripts/build_index.py  # Build FAISS index (run once)
   uvicorn api.main:app --reload  # Start API server
   streamlit run app.py           # Start UI (in a second terminal)
   ```

7. **Eval Results** — Table with columns: ID, Question (truncated), Intent ✓/✗, Language ✓/✗, Guardrail ✓/✗, Pass/Fail.

8. **Assumptions and Design Decisions**:
   - FAISS IndexFlatIP with L2-normalized vectors gives cosine similarity. Chosen for determinism and zero external dependencies.
   - Similarity threshold 0.40 was calibrated to be permissive for in-KB questions and strict for trap questions. Adjust in `guardrail.py`.
   - Conversation history is in-memory (per process). Production use would replace `ConversationManager` with Redis.
   - Language detection uses a heuristic-first approach (Unicode range for Devanagari, keyword matching for Hinglish) with no external API calls.
   - The KB is a single JSON file for simplicity. Production would use a database with FAISS persisted to disk (already implemented).

9. **Limitations**:
   - Room prices intentionally not in KB (per spec); any price question triggers guardrail or human handoff.
   - Language detection for very short Hinglish messages may classify as English; acceptable trade-off.
   - FAISS index must be rebuilt after KB updates via `python scripts/build_index.py`.

---

## Step 16 — scripts/build_index.py

```python
"""One-time script to build FAISS index from hotel KB."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ingest import build_index

if __name__ == "__main__":
    build_index()
```

---

## Step 17 — .gitignore

```
.env
venv/
__pycache__/
*.pyc
faiss_index/
*.pkl
.DS_Store
```

---

## Final Checklist

Before calling the project done, verify every item:

- [ ] `python scripts/build_index.py` runs without error and creates `faiss_index/hotel.index`
- [ ] `uvicorn api.main:app --reload` starts cleanly
- [ ] `POST /chat` with `{"message": "What time is the pool open?"}` returns a correct grounded answer
- [ ] `POST /chat` with `{"message": "What is the price per night?"}` returns human handoff (NEVER a price)
- [ ] `POST /chat` with `{"message": "Send me a payment link"}` returns human handoff (NEVER a URL)
- [ ] `POST /chat` with a Hindi message returns a Hindi response
- [ ] `POST /chat` with a Hinglish message returns a Hinglish response
- [ ] Multi-turn conversation: second message referencing first message context works correctly
- [ ] `python eval/run_eval.py` shows ≥ 8/10 passing (all 3 trap questions MUST pass)
- [ ] `streamlit run app.py` shows the chat UI with intent/language/guardrail badges
- [ ] README is complete with setup instructions and eval results table
- [ ] No API keys are committed to the repository
