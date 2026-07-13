"""
Sexy Streamlit UI for the Mistral LangChain chatbot.

Run with:
    pip install streamlit langchain-mistralai python-dotenv
    streamlit run chat_app.py

Keep your .env (with MISTRAL_API_KEY) in the parent folder, same as your original script.
"""

from dotenv import load_dotenv
from pathlib import Path
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ---- env & model setup (same logic as your CLI script) ----
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(
    page_title="Mistral Chat",
    page_icon="✨",
    layout="centered",
)

# ---- custom CSS: dark, glassy, gradient chat bubbles ----
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #1a1a2e 0%, #0f0f1a 60%, #05050a 100%);
        color: #e8e8f0;
    }
    section[data-testid="stSidebar"] {
        background: #14141f;
        border-right: 1px solid #2a2a3d;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7f5af0, #2cb67d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #8a8aa3;
        font-size: 0.9rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 16px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        border: 1px solid #2a2a3d;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, #2b1f4d, #1a1330);
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background: linear-gradient(135deg, #0d2b26, #0a1a1f);
    }
    /* Chat input — cover the outer container, the wrapper, and the textarea */
    div[data-testid="stChatInput"] {
        background: #1c1c2e !important;
        border-radius: 14px !important;
        border: 1px solid #3a3a55 !important;
        padding: 2px 4px !important;
    }
    div[data-testid="stChatInput"] > div {
        background: transparent !important;
        border: none !important;
    }
    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #e8e8f0 !important;
        caret-color: #7f5af0 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #6f6f8a !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border: 1px solid #7f5af0 !important;
        box-shadow: 0 0 0 2px rgba(127, 90, 240, 0.25) !important;
    }
    /* The bottom bar Streamlit wraps the chat input in */
    div[data-testid="stBottom"] > div {
        background: transparent !important;
    }
    div[data-testid="stBottomBlockContainer"] {
        background: linear-gradient(180deg, rgba(15,15,26,0) 0%, #0f0f1a 40%) !important;
    }
    /* Send button */
    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #7f5af0, #2cb67d) !important;
        border-radius: 10px !important;
        border: none !important;
    }
    div[data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- sidebar controls ----
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    system_prompt = st.text_area(
        "System prompt",
        value="You are a funny ai assistant.",
        height=100,
    )
    model_name = st.selectbox(
        "Model",
        ["mistral-small-2506", "mistral-large-latest", "mistral-medium-latest"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = [SystemMessage(content=system_prompt)]
        st.rerun()

st.markdown('<p class="app-title">✨ Mistral Chat</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">LangChain + Mistral, now with an actual face</p>', unsafe_allow_html=True)

# ---- session state ----
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=system_prompt)]

# keep system prompt in sync if user edits it mid-session
if isinstance(st.session_state.messages[0], SystemMessage):
    st.session_state.messages[0] = SystemMessage(content=system_prompt)

@st.cache_resource(show_spinner=False)
def get_model(name: str, temp: float):
    return ChatMistralAI(model=name, temperature=temp)

model = get_model(model_name, temperature)

# ---- render history (skip system message) ----
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# ---- chat input ----
prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)
        st.markdown(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))