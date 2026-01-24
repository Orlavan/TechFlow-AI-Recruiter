"""
TechFlow AI Recruiter - Streamlit Web Interface

A user-friendly chat interface for the recruitment bot.
Run with: streamlit run streamlit_app/streamlit_main.py
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.modules.agents import RecruitmentBot


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="TechFlow AI Recruiter",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    h1 {
        color: #00e5ff !important;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("TechFlow Solutions")
    st.markdown("### 🤖 AI Recruitment Assistant")
    st.markdown("---")
    st.info("**Currently hiring:**\nSenior Python Developer")
    st.markdown("---")

    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        st.session_state.bot = RecruitmentBot()
        st.rerun()


# =============================================================================
# MAIN INTERFACE
# =============================================================================

st.title("🤖 TechFlow AI Recruiter")
st.markdown("*Your AI-powered recruitment assistant*")
st.markdown("---")

# Initialize Bot
if "bot" not in st.session_state:
    try:
        st.session_state.bot = RecruitmentBot()
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
    greeting = (
        "Hello! I'm Alex from TechFlow Solutions. "
        "Thanks for your application! I'd love to chat about your experience with Python. "
        "How many years have you been working in the field?"
    )
    st.session_state.messages.append({"role": "assistant", "content": greeting})

# Display Chat History
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Handle User Input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(prompt)

    # Build history string
    history = "\n".join([
        f"{'Recruiter' if m['role']=='assistant' else 'Candidate'}: {m['content']}"
        for m in st.session_state.messages
    ])

    # Get bot response
    with st.spinner("Thinking..."):
        try:
            response, action = st.session_state.bot.process_turn(prompt, history)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant", avatar="🤖"):
                st.write(response)

            if action == "END":
                st.balloons()
                st.success("✓ Conversation completed. Thank you!")

        except Exception as e:
            st.error(f"Error: {e}")
