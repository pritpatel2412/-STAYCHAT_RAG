"""Streamlit frontend for the hotel RAG bot."""

import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/chat"
HEALTH_URL = "http://localhost:8000/health"

# Set up page configurations with a premium aesthetic
st.set_page_config(
    page_title="The Grand Hotel — Concierge",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FF8C00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .sidebar-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #FFD700;
        font-size: 1.4rem;
        margin-bottom: 20px;
    }
    
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        margin-right: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .badge-intent {
        background-color: rgba(0, 191, 255, 0.15);
        color: #00BFFF;
        border: 1px solid rgba(0, 191, 255, 0.3);
    }
    
    .badge-lang {
        background-color: rgba(186, 85, 211, 0.15);
        color: #BA55D3;
        border: 1px solid rgba(186, 85, 211, 0.3);
    }
    
    .badge-guard-off {
        background-color: rgba(50, 205, 50, 0.15);
        color: #32CD32;
        border: 1px solid rgba(50, 205, 50, 0.3);
    }
    
    .badge-guard-on {
        background-color: rgba(220, 20, 60, 0.15);
        color: #DC143C;
        border: 1px solid rgba(220, 20, 60, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .suggested-pill {
        display: inline-block;
        padding: 6px 14px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 5px;
    }
    
    .suggested-pill:hover {
        background-color: rgba(255, 215, 0, 0.15);
        border: 1px solid #FFD700;
        color: #FFD700;
        transform: translateY(-2px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize Message History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar panel
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏨 StayChat System</div>', unsafe_allow_html=True)
    st.caption("Robust ground-checked AI agent powering hotel guest services at scale.")
    st.markdown("---")

    # Display System Health
    st.markdown("### ⚙️ Component Status")
    try:
        health_resp = requests.get(HEALTH_URL, timeout=3)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            is_healthy = health_data.get("status") == "ok"
            
            # Gemini status
            gemini_status = "💚 Connected" if health_data["details"]["gemini_api"] == "healthy" else "🔴 Key Missing / Error"
            st.write(f"**Gemini API:** {gemini_status}")
            
            # FAISS DB status
            vector_status = "💚 Operational" if health_data["details"]["vector_index"] == "healthy" else "🔴 Index Not Found"
            st.write(f"**Vector Store:** {vector_status}")
        else:
            st.warning("⚠️ API Service is starting or unhealthy.")
    except Exception:
        st.error("🔴 API Offline (Run `uvicorn api.main:app` in terminal)")

    st.markdown("---")
    
    # Active Session Details
    st.markdown("### 🔑 Session Manager")
    st.write(f"**Active UUID:** `{st.session_state.session_id[:8]}...`")
    
    if st.button("🔄 Clear & Start Fresh Chat", use_container_width=True):
        try:
            # Call API to clear session
            requests.delete(f"http://localhost:8000/session/{st.session_state.session_id}", timeout=5)
        except Exception:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# Layout Configuration
col_chat, col_telemetry = st.columns([5, 2])

# Left column: Interactive Chat
with col_chat:
    st.markdown('<h1 class="main-title">🏨 The Grand Hotel</h1>', unsafe_allow_html=True)
    st.caption("Your 24/7 Grounded Concierge Assistant. Highly fluent in English, Hindi, and Hinglish.")
    
    # Suggested Questions Pills
    st.write("✨ **Quick Test Queries (Click copy-pastes to input below):**")
    pills = [
        "What time does the swimming pool open?",
        "What is the nightly price of a Deluxe Room? (Price Trap)",
        "Can you send a link to pay/book a suite? (URL Trap)",
        "Gym kitne baje tak khula rehta hai aur kya personal trainer milega?",
        "क्या होटल में पार्किंग उपलब्ध है? कितना शुल्क है?"
    ]
    
    # Render pill buttons
    for pill in pills:
        st.markdown(f"`{pill}`")

    st.markdown("<br>", unsafe_allow_html=True)

    # Render conversational chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Active chat input
    if prompt := st.chat_input("Ask me about amenities, dining, parking, rules, or policies..."):
        # Append and render guest query immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Trigger Concierge generation
        with st.chat_message("assistant"):
            with st.spinner("Grand Concierge is reviewing knowledge base..."):
                try:
                    resp = requests.post(API_URL, json={
                        "session_id": st.session_state.session_id,
                        "message": prompt
                    }, timeout=25)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown(data["response"])

                        # Save telemetry data in session state for right panel
                        st.session_state.last_telemetry = {
                            "intent": data["intent"],
                            "language": data["language"],
                            "guardrail_triggered": data["guardrail_triggered"],
                            "retrieved_docs": data["retrieved_docs"]
                        }

                        # Append assistant message to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["response"],
                            "meta": st.session_state.last_telemetry
                        })
                        
                        # Trigger UI refresh to render telemetry column update
                        st.rerun()
                    else:
                        st.error(f"Error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI server: {e}")

# Right column: Telemetry Dashboard & Debug Info
with col_telemetry:
    st.markdown("### 📊 Agent Telemetry")
    st.caption("Live decision metrics and semantic checks computed on the latest message.")
    
    if "last_telemetry" in st.session_state:
        tel = st.session_state.last_telemetry
        
        # Color coding guardrail status
        guard_class = "badge-guard-on" if tel["guardrail_triggered"] else "badge-guard-off"
        guard_label = "🚫 TRIGGERED" if tel["guardrail_triggered"] else "✅ ACTIVE & SAFE"
        
        # Custom badges
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="margin-bottom:12px;"><strong>Intent:</strong> <span class="badge badge-intent">{tel['intent'].upper()}</span></div>
                <div style="margin-bottom:12px;"><strong>Language:</strong> <span class="badge badge-lang">{tel['language'].upper()}</span></div>
                <div><strong>Guardrail:</strong> <span class="badge {guard_class}">{guard_label}</span></div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Vector retrieval logs
        st.markdown("### 🔍 Retrieved Context")
        st.caption("Semantic FAISS database hits used to ground the LLM response.")
        
        if tel["retrieved_docs"]:
            for i, doc in enumerate(tel["retrieved_docs"]):
                score = doc["score"]
                # Similarity grading color
                score_color = "#32CD32" if score >= 0.40 else "#FF4500"
                
                st.markdown(
                    f"""
                    <div style="font-size:0.85rem; border-left: 3px solid {score_color}; padding-left:10px; margin-bottom:12px;">
                        <strong>Rank {i+1}: {doc['title']}</strong><br/>
                        <span style="color:#888;">Doc ID: <code>{doc['id']}</code></span><br/>
                        <span style="color:{score_color}; font-weight:600;">Similarity: {score:.4f}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.info("No semantic retrieval logs for the latest turn.")
    else:
        st.info("Start chatting to view real-time vector & guardrail telemetry.")
