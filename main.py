"""
TechFlow AI Recruiter - Main Entry Point

A Multi-Agent Recruitment Chatbot for screening candidates,
answering questions via RAG, and scheduling interviews.

Usage:
    Console: python main.py
    Web UI:  streamlit run streamlit_app/streamlit_main.py
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from app.modules.agents import RecruitmentBot
from app.modules.info_agent.ingest import EmbeddingsManager


# =============================================================================
# SETUP
# =============================================================================

def setup():
    """Initialize the system components (embeddings, database, etc.)."""
    print("Initializing TechFlow AI Recruiter...")

    try:
        manager = EmbeddingsManager()
        _ = manager.vectorstore
        print("✓ Embeddings loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize embeddings: {e}")

    print("✓ System ready!\n")


# =============================================================================
# CONSOLE CHAT INTERFACE
# =============================================================================

def run_console_chat():
    """Run the interactive console chat interface."""

    print("=" * 60)
    print("       TechFlow AI Recruiter")
    print("=" * 60)
    print("\nCommands: 'quit' to exit | 'reset' to restart\n")
    print("-" * 60)

    # Initialize bot
    bot = RecruitmentBot()

    # Initial greeting
    greeting = (
        "Hi! I'm Alex from TechFlow Solutions. "
        "Thanks for your interest in our Python Developer position. "
        "Can you tell me about your experience with Python?"
    )
    print(f"\nRecruiter: {greeting}\n")
    history = f"Recruiter: {greeting}"

    # Main conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ['quit', 'exit']:
            print("\nThank you for chatting! Goodbye!")
            break

        if user_input.lower() == 'reset':
            bot.reset()
            history = f"Recruiter: {greeting}"
            print(f"\n{'='*60}\n--- New Conversation ---\n{'='*60}")
            print(f"\nRecruiter: {greeting}\n")
            continue

        # Process the conversation turn
        try:
            response, action = bot.process_turn(user_input, history)
            history += f"\nCandidate: {user_input}\nRecruiter: {response}"

            print(f"\nRecruiter: {response}")
            print(f"[{action}]\n")

            if action == "END":
                print("-" * 60)
                print("Conversation ended. Type 'reset' to start a new one.")
                print("-" * 60 + "\n")

        except Exception as e:
            print(f"\n⚠ Error: {e}\nPlease try again.\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""

    # Validate API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not found.")
        print("Please create a .env file with your API key.")
        sys.exit(1)

    setup()
    run_console_chat()


if __name__ == "__main__":
    main()
