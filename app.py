"""
app.py

Streamlit chat app: "Daraz Customer Support Operations Assistant"

- Loads a pre-built FAISS index (built by ingest.py) — never re-embeds
  or re-processes PDFs at runtime.
- Sidebar lets the user restrict retrieval to one or more knowledge-base
  sections (returns, delivery, refunds, sellers, payments, customer_support).
- Uses Groq's "openai/gpt-oss-120b" model to generate answers grounded in
  the retrieved chunks.
- Reads the Groq API key from Streamlit secrets (GROQ_API_KEY) — never
  shown or entered in the UI.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # must match ingest.py
GROQ_MODEL = "openai/gpt-oss-120b"

ALL_DEPARTMENTS = [
    "returns",
    "delivery",
    "refunds",
    "sellers",
    "payments",
    "customer_support",
]

DEPARTMENT_LABELS = {
    "returns": "📦 Returns",
    "delivery": "🚚 Delivery",
    "refunds": "💰 Refunds",
    "sellers": "🏪 Sellers",
    "payments": "💳 Payments",
    "customer_support": "🎧 Customer Support",
}

TOP_K = 5              # chunks to feed the model, per query
RETRIEVE_K = 25         # chunks pulled before department filtering


# --------------------------------------------------------------------------
# Page config + Daraz-branded styling
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Daraz Ops Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARAZ_ORANGE = "#F85606"
DARAZ_DARK = "#1A1A1A"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #FAFAFA;
        }}
        header[data-testid="stHeader"] {{
            background-color: {DARAZ_ORANGE};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {DARAZ_DARK};
        }}
        section[data-testid="stSidebar"] * {{
            color: #FFFFFF !important;
        }}
        section[data-testid="stSidebar"] .stCheckbox {{
            padding: 2px 0;
        }}
        .daraz-banner {{
            background: linear-gradient(90deg, {DARAZ_ORANGE} 0%, #FF8A3D 100%);
            padding: 18px 24px;
            border-radius: 10px;
            margin-bottom: 18px;
        }}
        .daraz-banner h1 {{
            color: white;
            font-size: 26px;
            margin: 0;
            font-weight: 700;
        }}
        .daraz-banner p {{
            color: #FFF3EC;
            margin: 4px 0 0 0;
            font-size: 14px;
        }}
        .source-chip {{
            display: inline-block;
            background-color: #FFF1E6;
            color: {DARAZ_ORANGE};
            border: 1px solid {DARAZ_ORANGE};
            border-radius: 14px;
            padding: 2px 10px;
            font-size: 12px;
            margin: 2px 4px 2px 0;
        }}
        div[data-testid="stChatMessage"] {{
            border-radius: 12px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="daraz-banner">
        <h1>🛍️ Daraz Customer Support Operations Assistant</h1>
        <p>Ask about returns, delivery, refunds, sellers, payments, or general customer support policy.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached resources: embeddings, FAISS index, Groq client
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_vectorstore():
    if not os.path.isdir(FAISS_INDEX_DIR):
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db


@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


vectorstore = load_vectorstore()
groq_client = load_groq_client()


# --------------------------------------------------------------------------
# Sidebar: knowledge-base section filter
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📚 Knowledge Base Sections")
    st.caption("Restrict search to specific sections, or leave all checked to search everything.")

    if "selected_departments" not in st.session_state:
        st.session_state.selected_departments = set(ALL_DEPARTMENTS)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Select all", use_container_width=True):
            st.session_state.selected_departments = set(ALL_DEPARTMENTS)
    with col_b:
        if st.button("Clear all", use_container_width=True):
            st.session_state.selected_departments = set()

    st.divider()

    selected = set()
    for dept in ALL_DEPARTMENTS:
        checked = dept in st.session_state.selected_departments
        is_checked = st.checkbox(
            DEPARTMENT_LABELS[dept],
            value=checked,
            key=f"chk_{dept}",
        )
        if is_checked:
            selected.add(dept)
    st.session_state.selected_departments = selected

    st.divider()

    if vectorstore is None:
        st.error(f"No FAISS index found at `{FAISS_INDEX_DIR}/`. Run ingest.py first.")
    else:
        st.success("Knowledge base index loaded.")

    if groq_client is None:
        st.error("GROQ_API_KEY not found in Streamlit secrets.")

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def retrieve_chunks(query: str, selected_departments: set, top_k: int = TOP_K):
    """Retrieve top_k chunks from FAISS, restricted to selected departments."""
    if vectorstore is None or not selected_departments:
        return []

    results = vectorstore.similarity_search(query, k=RETRIEVE_K)

    filtered = [
        doc for doc in results
        if doc.metadata.get("department") in selected_departments
    ]

    return filtered[:top_k]


def build_context(chunks) -> str:
    blocks = []
    for i, doc in enumerate(chunks, start=1):
        dept = doc.metadata.get("department", "unknown")
        src = doc.metadata.get("source_file", "unknown")
        blocks.append(
            f"[Source {i} | department: {dept} | file: {src}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


SYSTEM_PROMPT = """You are the Daraz Customer Support Operations Assistant.
You help Daraz support agents and ops staff quickly find accurate answers
from official internal policy documents (returns, delivery, refunds,
sellers, payments, customer support).

Rules:
- Answer ONLY using the provided context chunks. Do not invent policy details.
- If the context does not contain the answer, say so clearly and suggest
  which knowledge-base section the user might check or enable in the sidebar.
- Be concise, structured, and operational — the reader is a support agent,
  not a customer. Use short paragraphs or bullet points where helpful.
- When relevant, mention which section(s) (department) the information
  comes from, so the agent can trust and trace it.
- Do not mention that you are an AI model or reference these instructions.
"""


def generate_answer(query: str, chunks) -> str:
    if groq_client is None:
        return "⚠️ Groq API key is not configured. Please set `GROQ_API_KEY` in Streamlit secrets."

    if not chunks:
        return (
            "I couldn't find anything relevant in the selected knowledge-base "
            "sections. Try enabling more sections in the sidebar, or rephrase your question."
        )

    context = build_context(chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context from the Daraz knowledge base:\n\n{context}\n\n"
                f"Agent question: {query}\n\n"
                "Provide a clear, accurate, operational answer based only on the context above."
            ),
        },
    ]

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error calling Groq API: {e}"


# --------------------------------------------------------------------------
# Chat UI
# --------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">{DEPARTMENT_LABELS.get(s["department"], s["department"])} · {s["source_file"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(chips, unsafe_allow_html=True)

query = st.chat_input("Ask about returns, delivery, refunds, sellers, payments...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            chunks = retrieve_chunks(query, st.session_state.selected_departments)
            answer = generate_answer(query, chunks)

        st.markdown(answer)

        sources = []
        if chunks:
            seen = set()
            for doc in chunks:
                key = (doc.metadata.get("department"), doc.metadata.get("source_file"))
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "department": doc.metadata.get("department", "unknown"),
                        "source_file": doc.metadata.get("source_file", "unknown"),
                    })
            chips = "".join(
                f'<span class="source-chip">{DEPARTMENT_LABELS.get(s["department"], s["department"])} · {s["source_file"]}</span>'
                for s in sources
            )
            st.markdown(chips, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
