import json
import streamlit as st
import requests
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Gita AI — Philosophical Mentor",
    page_icon="🪷",
    layout="centered"
)

st.title("🪷 Gita AI")
st.caption("Timeless wisdom from the Bhagavad Gita for modern dilemmas.")

# Check for Gemini API Key in Streamlit Secrets
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing `GEMINI_API_KEY`. Please set it in Streamlit App Settings -> Secrets.")
    st.stop()

api_key = st.secrets["GEMINI_API_KEY"]

@st.cache_resource(show_spinner="Preparing Bhagavad Gita scripture database...")
def init_system():
    # 1. Fetch official open-source Gita verses
    url = "https://raw.githubusercontent.com/vedicscriptures/bhagavad-gita-data/master/slok.json"
    resp = requests.get(url, timeout=30)
    raw_data = resp.json()

    clean_verses = []
    for item in raw_data:
        chapter = item.get("chapter")
        verse = item.get("verse")
        sloka = item.get("slok", "")
        transliteration = item.get("transliteration", "")
        
        translation = ""
        if "siva" in item and "et" in item["siva"]:
            translation = item["siva"]["et"]
        elif "tej" in item and "et" in item["tej"]:
            translation = item["tej"]["et"]

        if chapter and verse and translation:
            clean_verses.append({
                "id": f"bg_{chapter}_{verse}",
                "chapter": int(chapter),
                "verse": int(verse),
                "sloka": sloka.strip(),
                "transliteration": transliteration.strip(),
                "translation": translation.strip()
            })

    # 2. Initialize in-memory Vector Database
    chroma_client = chromadb.Client()
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.create_collection(
        name="bhagavad_gita",
        embedding_function=emb_fn
    )

    # 3. Index verses
    documents = [f"Chapter {v['chapter']}, Verse {v['verse']}: {v['translation']}" for v in clean_verses]
    metadatas = clean_verses
    ids = [v["id"] for v in clean_verses]

    # Batch insert to stay memory efficient
    for i in range(0, len(clean_verses), 100):
        collection.add(
            documents=documents[i:i+100],
            metadatas=metadatas[i:i+100],
            ids=ids[i:i+100]
        )

    # 4. Initialize Gemini Client
    ai_client = genai.Client(api_key=api_key)
    return collection, ai_client

collection, ai_client = init_system()

SYSTEM_PROMPT = """
You are a calm, compassionate life mentor grounded in the Bhagavad Gita.
When a user asks for guidance on a life problem:
1. Empathy: Warmly acknowledge their emotional state in 1-2 sincere sentences.
2. Core Verse: Identify the most fitting verse from the provided context. State Chapter and Verse, provide the transliteration, and give a clear English translation.
3. Philosophical Insight: Explain how the Gita's philosophy applies directly to their modern dilemma.
4. Action Step: Provide one tangible, actionable practice they can execute today.
Keep your tone peer-like, wise, and supportive. Avoid dogmatic or overly archaic language.
"""

def get_mentor_response(user_query: str):
    # Vector semantic search
    search = collection.query(query_texts=[user_query], n_results=3)
    retrieved_meta = search["metadatas"][0]

    context_segments = []
    for item in retrieved_meta:
        context_segments.append(
            f"BG {item['chapter']}.{item['verse']}:\n"
            f"Transliteration: {item['transliteration']}\n"
            f"Translation: {item['translation']}\n"
        )
    context_str = "\n---\n".join(context_segments)

    prompt = f"""
    User's Dilemma:
    "{user_query}"

    Retrieved Verses from Bhagavad Gita:
    {context_str}

    Provide your counsel following your system instructions.
    """

    response = ai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.6,
        )
    )
    return response.text, retrieved_meta

# Chat State Management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Namaste. What situation or decision is weighing on your mind today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("E.g., I worked hard but failed, how do I deal with disappointment?"):
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
