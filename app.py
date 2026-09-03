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
st.markdown("""
<style>
/* Main app background: cosmic gradient with atmospheric divine vignette */
.stApp {
    background: radial-gradient(circle at 50% 15%, rgba(26, 35, 126, 0.45) 0%, rgba(10, 15, 30, 0.95) 75%),
                linear-gradient(180deg, #070b14 0%, #0d1527 50%, #050811 100%);
    color: #e0e6ed;
    font-family: 'Inter', system-ui, sans-serif;
}

/* Header styling */
.gita-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    background: linear-gradient(135deg, #ffd700 0%, #ffae19 50%, #fff2a3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    text-shadow: 0 0 25px rgba(255, 215, 0, 0.25);
}

.gita-subtitle {
    text-align: center;
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 2rem;
    font-style: italic;
}

/* Translucent Glassmorphism Chat Bubbles */
[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 215, 0, 0.15) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    margin-bottom: 1rem !important;
}

/* Floating Golden Aura for Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #ffae19 !important;
    box-shadow: 0 0 20px rgba(255, 174, 25, 0.08) !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    background: rgba(13, 21, 39, 0.85) !important;
    border: 1px solid rgba(255, 215, 0, 0.3) !important;
    border-radius: 24px !important;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.1) !important;
}

/* Expander panel styling for verses */
.streamlit-expanderHeader {
    background: rgba(20, 30, 55, 0.5) !important;
    border: 1px solid rgba(255, 215, 0, 0.2) !important;
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
