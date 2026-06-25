"""
Memora UI — polished Streamlit frontend.
Chat with the agent and watch its memory work live:
what it stores, updates, forgets, and recalls.
Talks to the FastAPI backend over HTTP.
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

# ---- Page setup ----
st.set_page_config(page_title="Memora", page_icon="🧠", layout="wide")

# ---- Light custom styling ----
st.markdown("""
<style>
    .main-title { font-size: 2.4rem; font-weight: 800; color: #6C4CE0; margin-bottom: 0; }
    .subtitle { color: #888; font-size: 1rem; margin-top: 0; }
    .mem-card {
        background: #f7f5ff; border-left: 4px solid #6C4CE0;
        padding: 10px 14px; border-radius: 8px; margin-bottom: 8px;
    }
    .action-added { color: #1a8a4a; font-weight: 600; }
    .action-updated { color: #c47f00; font-weight: 600; }
    .action-skipped { color: #888; font-weight: 600; }
    .imp-badge {
        background: #6C4CE0; color: white; padding: 1px 8px;
        border-radius: 10px; font-size: 0.75rem; margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<p class="main-title">🧠 Memora</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A self-managing memory agent — it remembers what matters, forgets what doesn\'t, and resolves contradictions automatically.</p>', unsafe_allow_html=True)
st.divider()

# ---- Session state (keeps chat history across reruns) ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_actions" not in st.session_state:
    st.session_state.last_actions = []

# ---- Sidebar: user selection + decay control ----
with st.sidebar:
    st.header("Settings")
    user_id = st.text_input("User ID", value="demo_user")
    st.caption("Each user has separate memories.")
    st.divider()
    if st.button("🧹 Run Decay (forget stale)"):
        try:
            resp = requests.post(f"{API_URL}/decay/{user_id}", timeout=30)
            summary = resp.json()
            st.success(f"Checked {summary['checked']}, forgot {summary['pruned']}.")
            if summary["pruned_texts"]:
                for t in summary["pruned_texts"]:
                    st.caption(f"Forgot: {t}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")

# ---- Two-column layout: chat | memory panel ----
col_chat, col_mem = st.columns([3, 2])

# ===== LEFT: Chat =====
with col_chat:
    st.subheader("💬 Conversation")

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Show what the agent did with memory on the last turn
    if st.session_state.last_actions:
        with st.expander("🔍 What Memora did with memory this turn", expanded=True):
            for a in st.session_state.last_actions:
                cls = f"action-{a['action']}"
                st.markdown(f"<span class='{cls}'>● {a['action'].upper()}</span> — {a['detail']}", unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Say something to Memora...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={"user_id": user_id, "message": prompt},
                timeout=60,
            )
            data = resp.json()
            st.session_state.messages.append({"role": "assistant", "content": data["reply"]})
            st.session_state.last_actions = data.get("memory_actions", [])
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Could not reach API: {e}"})
            st.session_state.last_actions = []
        st.rerun()

# ===== RIGHT: Memory panel =====
with col_mem:
    st.subheader("🧠 Stored Memories")
    try:
        resp = requests.get(f"{API_URL}/memories/{user_id}", timeout=30)
        data = resp.json()
        st.caption(f"{data['count']} memories stored for **{user_id}**")

        for m in data["memories"]:
            imp = m["metadata"].get("importance", 5)
            mtype = m["metadata"].get("mem_type", "other")
            st.markdown(
                f"<div class='mem-card'>{m['text']}"
                f"<span class='imp-badge'>imp {imp}</span><br>"
                f"<small style='color:#999'>{mtype}</small></div>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Could not load memories: {e}")