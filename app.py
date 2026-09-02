import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from gita_data import GITA_VERSES

st.set_page_config(
    page_title="Gita AI — Philosophical Mentor",
    page_icon="🪷",
    layout="centered"
)

st.title("🪷 Gita AI")
st.caption("Timeless wisdom from the Bhagavad Gita applied to modern life dilemmas.")

# Check for Groq API key in Streamlit secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error("Missing `GROQ_API_KEY`. Please set it in Streamlit: Manage app -> App settings -> Secrets.")
    st.stop()

api_key = str(st.secrets["GROQ_API_KEY"]).strip().replace('"', '').replace("'", "")

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

    client = Groq(api_key=api_key)
    return collection, client

collection, groq_client = init_system()

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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User Dilemma: \"{user_query}\"\n\nRetrieved Gita Verses:\n{context_str}\n\nProvide your counsel:"}
    ]

    try:
        # Fetch all models and filter strictly for standard chat/instruct models
        all_models = [m.id for m in groq_client.models.list().data]
        
        # Priority list of reliable general-purpose text models
        preferred_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]

        target_model = None
        for cand in preferred_models:
            if cand in all_models:
                target_model = cand
                break

        # If none from preferred list match, pick the first Llama or Mixtral model only
        if not target_model:
            chat_candidates = [
                m for m in all_models 
                if ("llama" in m.lower() or "mixtral" in m.lower() or "gemma" in m.lower())
                and "whisper" not in m.lower()
                and "/" not in m  # skips gated third-party models like canopylabs/
            ]
            target_model = chat_candidates[0] if chat_candidates else all_models[0]

        completion = groq_client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=0.6,
        )
        return completion.choices[0].message.content, retrieved_meta
    except Exception as e:
        return f"⚠️ **API Request Error**: {str(e)}", retrieved_meta

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
