import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from gita_data import GITA_VERSES

st.set_page_config(
    page_title="BhagavanGPT — Divine Mentor",
    page_icon="🪷",
    layout="centered"
)

# --- CINEMATIC VISUAL STYLING (CSS / CGI THEME) ---
# --- CINEMATIC MAHAVATAR VISHNU BACKDROP & ORIGINAL TITLES ---
st.markdown("""
<style>
/* 1. Full-screen Cinematic App Base */
.stApp {
    background-color: #06070c;
    color: #e2e8f0;
    font-family: 'Inter', system-ui, sans-serif;
}

/* 2. Mahavatar Vishnu Background Layer */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-image: 
        radial-gradient(circle at 50% 30%, rgba(6, 7, 12, 0.4) 0%, rgba(6, 7, 12, 0.88) 85%),
        url("https://images.unsplash.com/photo-1599827552599-eadf5e0a9695?q=80&w=1600&auto=format&fit=crop");
    background-size: cover;
    background-position: center top;
    background-repeat: no-repeat;
    opacity: 0.32;
    z-index: 0;
    pointer-events: none;
    filter: contrast(125%) brightness(85%);
}

/* 3. Golden Cosmic Aura behind the Avatar */
.stApp::after {
    content: "";
    position: fixed;
    top: 15%;
    left: 50%;
    transform: translateX(-50%);
    width: 550px;
    height: 550px;
    background: radial-gradient(circle, rgba(255, 174, 25, 0.2) 0%, rgba(255, 100, 0, 0.08) 45%, transparent 70%);
    border-radius: 50%;
    z-index: 0;
    pointer-events: none;
    filter: blur(50px);
    animation: divineGlow 7s ease-in-out infinite alternate;
}

@keyframes divineGlow {
    0% { transform: translateX(-50%) scale(0.95); opacity: 0.6; }
    100% { transform: translateX(-50%) scale(1.15); opacity: 1; }
}

/* 4. Original Sanskrit Title & Subtitle Styling */
.gita-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    background: linear-gradient(135deg, #ffd700 0%, #ffae19 50%, #fff2a3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    text-shadow: 0 0 25px rgba(255, 215, 0, 0.3);
    position: relative;
    z-index: 1;
}

.gita-subtitle {
    text-align: center;
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 2rem;
    font-style: italic;
    position: relative;
    z-index: 1;
}

/* 5. Glassmorphic Chat Bubbles */
[data-testid="stChatMessage"] {
    background: rgba(12, 16, 28, 0.72) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(255, 215, 0, 0.18) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6) !important;
    margin-bottom: 1rem !important;
    position: relative;
    z-index: 1;
}

/* Gold Accent on Mentor Responses */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #ffae19 !important;
    box-shadow: 0 0 20px rgba(255, 174, 25, 0.12) !important;
}

/* 6. Chat Input */
[data-testid="stChatInput"] {
    background: rgba(13, 19, 35, 0.88) !important;
    border: 1px solid rgba(255, 215, 0, 0.3) !important;
    border-radius: 24px !important;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.12) !important;
}

/* 7. Shloka Expander Card */
.streamlit-expanderHeader {
    background: rgba(18, 24, 42, 0.6) !important;
    border: 1px solid rgba(255, 215, 0, 0.25) !important;
    border-radius: 12px !important;
    color: #ffd700 !important;
}
</style>

<div class="gita-title">🪷 श्रीमद्भगवद्गीता AI</div>
<div class="gita-subtitle">"Whenever the mind wanders, bring it gently back to the Self."</div>
""", unsafe_allow_html=True)
# Check for Groq API key in Streamlit secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error("Missing `GROQ_API_KEY`. Please set it in Streamlit: Manage app -> App settings -> Secrets.")
    st.stop()

api_key = str(st.secrets["GROQ_API_KEY"]).strip().replace('"', '').replace("'", "")

@st.cache_resource(show_spinner="Connecting to the scripture repository...")
def init_system():
    chroma_client = chromadb.Client()
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    
    collection = chroma_client.get_or_create_collection(
        name="bhagavad_gita",
        embedding_function=emb_fn
    )

    # Only populate if the collection is empty (prevents duplicate ID errors on reload)
    if collection.count() == 0:
        documents = [
            f"Chapter {v['chapter']}, Verse {v['verse']}. Themes: {v['theme']}. Meaning: {v['translation']}"
            for v in GITA_VERSES
        ]
        metadatas = GITA_VERSES
        ids = [v["id"] for v in GITA_VERSES]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    documents = [
        f"Chapter {v['chapter']}, Verse {v['verse']}. Themes: {v['theme']}. Meaning: {v['translation']}"
        for v in GITA_VERSES
    ]
    metadatas = GITA_VERSES
    ids = [v["id"] for v in GITA_VERSES]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    client = Groq(api_key=api_key)
    return collection, client

collection, groq_client = init_system()

SYSTEM_PROMPT = """
You are a calm, enlightened life mentor speaking with the timeless wisdom and compassionate authority of the Bhagavad Gita (like Krishna guiding Arjuna on the battlefield).
When a user shares a life dilemma:
1. Empathy: Warmly acknowledge their inner state in 1-2 grounding sentences.
2. Core Verse: Identify the most fitting verse from the retrieved context. State Chapter and Verse, Sanskrit transliteration, and English translation cleanly.
3. Philosophical Insight: Explain how the Gita's philosophy applies directly to their modern dilemma (e.g., detachment from results, calming the restless mind, rising above grief).
4. Practical Step: Give one simple, concrete action step or mindset exercise they can do today.
Keep your tone serene, encouraging, and clear. Avoid preachy or archaic language.
"""

def get_mentor_response(user_query: str):
    search = collection.query(query_texts=[user_query], n_results=4)
    retrieved_meta = search["metadatas"][0]

    context_segments = []
    for item in retrieved_meta:
        context_segments.append(
            f"BG {item['chapter']}.{item['verse']} (Themes: {item['theme']}):\n"
            f"Transliteration: {item['transliteration']}\n"
            f"Translation: {item['translation']}\n"
        )
    context_str = "\n---\n".join(context_segments)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User Dilemma: \"{user_query}\"\n\nCandidate Gita Verses:\n{context_str}\n\nProvide your counsel:"}
    ]

    candidate_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile"
    ]

    last_error = None
    for model_name in candidate_models:
        try:
            completion = groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.75,
            )
            return completion.choices[0].message.content, retrieved_meta
        except Exception as e:
            last_error = e
            continue

    return f"⚠️ **API Request Error**: {str(last_error)}", retrieved_meta

# Chat State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Namaste, friend. Cast aside your doubts and tell me: what storm is troubling your mind today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("Ask about purpose, fear of failure, heartbreak, focus..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Channeling the eternal wisdom..."):
            reply, matched_verses = get_mentor_response(prompt_input)
            st.markdown(reply)

            with st.expander("📖 View Retrieved Sacred Shlokas"):
                for verse in matched_verses:
                    st.markdown(f"✨ **Chapter {verse['chapter']}, Verse {verse['verse']}**")
                    st.markdown(f"*{verse['transliteration']}*")
                    st.info(f"\"{verse['translation']}\"")

    st.session_state.messages.append({"role": "assistant", "content": reply})
