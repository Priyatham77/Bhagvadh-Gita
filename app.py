import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types
from gita_data import GITA_VERSES

st.set_page_config(
    page_title="Gita AI — Philosophical Mentor",
    page_icon="🪷",
    layout="centered"
)

st.title("🪷 Gita AI")
st.caption("Timeless wisdom from the Bhagavad Gita applied to modern life dilemmas.")

# Check for API key in Streamlit secrets
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing `GEMINI_API_KEY`. Please set it in Streamlit: Manage app -> App settings -> Secrets.")
    st.stop()

api_key = st.secrets["GEMINI_API_KEY"]

@st.cache_resource(show_spinner="Initializing vector store with Gita wisdom...")
def init_system():
    # In-memory ChromaDB vector store
    chroma_client = chromadb.Client()
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    
    collection = chroma_client.create_collection(
        name="bhagavad_gita",
        embedding_function=emb_fn
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

    ai_client = genai.Client(api_key=api_key)
    return collection, ai_client

collection, ai_client = init_system()

SYSTEM_PROMPT = """
You are a grounded, deeply compassionate life mentor who draws wisdom directly from the Bhagavad Gita.
When a user asks for guidance on a personal problem:
1. Empathy: Warmly acknowledge their feelings in 1-2 sincere sentences.
2. Core Verse: Identify the best matching verse from the retrieved context. State Chapter and Verse, Sanskrit transliteration, and English translation.
3. Philosophical Application: Explain how this verse applies to their exact modern dilemma (e.g., detachment from results, calming the restless mind, rising above grief).
4. Practical Step: Give one simple, concrete action step or mindset exercise they can do today.
Keep your tone warm, wise, peer-like, and supportive. Avoid dogmatic or preachy language.
"""

def get_mentor_response(user_query: str):
    search = collection.query(query_texts=[user_query], n_results=2)
    retrieved_meta = search["metadatas"][0]

    context_segments = []
    for item in retrieved_meta:
        context_segments.append(
            f"BG {item['chapter']}.{item['verse']} (Themes: {item['theme']}):\n"
            f"Transliteration: {item['transliteration']}\n"
            f"Translation: {item['translation']}\n"
        )
    context_str = "\n---\n".join(context_segments)

    prompt = f"""
    User's Dilemma:
    "{user_query}"

    Retrieved Gita Verses:
    {context_str}

    Provide your counsel following your system instructions.
    """

    try:
        response = ai_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.6,
            )
        )
        return response.text, retrieved_meta
    except Exception as e:
        return f"⚠️ **API Request Error**: {str(e)}\n\nPlease verify that your `GEMINI_API_KEY` in Streamlit Cloud Secrets is valid and has access to `gemini-1.5-flash`.", retrieved_meta

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Namaste. What situation or challenge is on your mind today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("E.g., I'm terrified of failing my upcoming exams..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Contemplating the verses..."):
            reply, matched_verses = get_mentor_response(prompt_input)
            st.markdown(reply)

            with st.expander("📖 View Retrieved Source Verses"):
                for verse in matched_verses:
                    st.markdown(f"**Chapter {verse['chapter']}, Verse {verse['verse']}**")
                    st.markdown(f"*{verse['transliteration']}*")
                    st.info(f"\"{verse['translation']}\"")

    st.session_state.messages.append({"role": "assistant", "content": reply})
