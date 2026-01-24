"""
Main Application Entry Point
TechFlow AI Recruiter - Multi-Agent Recruitment Chatbot

Run this file to start the console-based chat interface.
For web interface, use: streamlit run streamlit_app/streamlit_main.py
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.modules.agents import RecruitmentBot
from app.modules.database import init_database, DB_PATH
from app.modules.embeddings import init_embeddings


def setup():
    """Initialize database and embeddings if needed."""
    print("Initializing system...")

    # Initialize database if not exists
    if not os.path.exists(DB_PATH):
        print("Creating database...")
        init_database()

    # Initialize embeddings (checks if exists internally)
    print("Checking embeddings...")
    try:
        from app.modules.embeddings import EmbeddingsManager
        manager = EmbeddingsManager()
        _ = manager.vectorstore  # This will create if not exists
        print("Embeddings ready.")
    except Exception as e:
        print(f"Warning: Could not initialize embeddings: {e}")

    print("System ready!\n")


def run_console_chat():
    """
    Runs the interactive console chat interface.
    """
    print("=" * 60)
    print("       TechFlow Solutions - AI Recruitment Assistant")
    print("=" * 60)
    print()
    print("Type 'quit' or 'exit' to end the conversation.")
    print("Type 'reset' to start a new conversation.")
    print("-" * 60)
    print()

    # Initialize bot
    bot = RecruitmentBot()

    # Initial greeting
    greeting = "Hi! I'm Alex from TechFlow Solutions. Thanks for your interest in our Python Developer position. Can you tell me about your experience with Python?"
    print(f"Recruiter: {greeting}")
    print()

    history = f"Recruiter: {greeting}"

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit']:
            print("\nThank you for chatting! Goodbye!")
            break

        if user_input.lower() == 'reset':
            bot.reset()
            history = f"Recruiter: {greeting}"
            print(f"\n--- New Conversation ---\n")
            print(f"Recruiter: {greeting}\n")
            continue

        # Process the turn
        try:
            response, action = bot.process_turn(user_input, history)
            history += f"\nCandidate: {user_input}\nRecruiter: {response}"

            print(f"\nRecruiter: {response}")
            print(f"[Action: {action}]\n")

            if action == "END":
                print("-" * 60)
                print("Conversation ended. Type 'reset' to start a new one.")
                print("-" * 60)

        except Exception as e:
            print(f"\nError processing message: {e}")
            print("Please try again.\n")


def main():
    """Main entry point."""
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key.")
        sys.exit(1)

    # Setup
    setup()

    # Run chat
    run_console_chat()


if __name__ == "__main__":
    main()
