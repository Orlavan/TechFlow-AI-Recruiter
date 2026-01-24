"""
Streamlit Utility Functions
Helper functions for the Streamlit UI components.
"""

import streamlit as st
from typing import List, Dict


def format_history(messages: List[Dict]) -> str:
    """
    Formats message list into history string for the bot.

    Args:
        messages: List of message dictionaries with 'role' and 'content'

    Returns:
        Formatted history string
    """
    history_parts = []
    for msg in messages:
        role = "Recruiter" if msg["role"] == "assistant" else "Candidate"
        history_parts.append(f"{role}: {msg['content']}")
    return "\n".join(history_parts)


def display_message(role: str, content: str):
    """
    Displays a chat message with appropriate styling.

    Args:
        role: 'user' or 'assistant'
        content: Message content
    """
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.write(content)


def show_conversation_stats(messages: List[Dict]):
    """
    Displays conversation statistics in the sidebar.
    """
    total_messages = len(messages)
    user_messages = sum(1 for m in messages if m["role"] == "user")
    bot_messages = total_messages - user_messages

    st.sidebar.markdown("### Conversation Stats")
    st.sidebar.write(f"Total turns: {total_messages}")
    st.sidebar.write(f"Your messages: {user_messages}")
    st.sidebar.write(f"Bot messages: {bot_messages}")


def get_custom_css() -> str:
    """
    Returns custom CSS for the Streamlit app.
    """
    return """
    <style>
        .stApp {
            background-color: #0e1117;
            color: #f0f2f6;
        }
        h1 {
            color: #00e5ff !important;
            font-weight: 700;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 10px;
        }
        div[data-testid="stChatMessage"]:nth-child(even) {
            background: linear-gradient(135deg, #2b313e 0%, #1e232e 100%);
            border-left: 4px solid #00e5ff;
        }
        div[data-testid="stChatMessage"]:nth-child(odd) {
            background: linear-gradient(135deg, #1e232e 0%, #171b24 100%);
            border-left: 4px solid #ff007f;
        }
        section[data-testid="stSidebar"] {
            background-color: #171b24;
        }
        .stButton>button {
            background: linear-gradient(90deg, #00e5ff 0%, #00aaff 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 229, 255, 0.4);
        }
    </style>
    """
