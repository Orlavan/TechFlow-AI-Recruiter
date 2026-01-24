"""
Exit Advisor Module
Handles conversation termination detection.
Supports Fine-Tuned model for improved exit detection.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Fine-tuned model ID (set in .env after fine-tuning)
FINETUNED_EXIT_MODEL = os.getenv("FINETUNED_EXIT_MODEL", None)
DEFAULT_MODEL = "gpt-3.5-turbo"


class ExitAdvisor:
    """
    Evaluates whether a conversation should end.
    Uses Fine-Tuned model when available for better accuracy.

    Scenarios for ending:
    - Candidate not interested
    - Candidate found another job
    - Interview successfully booked
    - Candidate requests to stop
    """

    def __init__(self, use_finetuned: bool = True):
        """
        Initialize Exit Advisor.

        Args:
            use_finetuned: Whether to use fine-tuned model if available
        """
        model_name = FINETUNED_EXIT_MODEL if (use_finetuned and FINETUNED_EXIT_MODEL) else DEFAULT_MODEL
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.is_finetuned = bool(FINETUNED_EXIT_MODEL and use_finetuned)

        # Few-shot examples for exit detection
        self.examples = [
            {"input": "I'm not interested anymore", "output": "END"},
            {"input": "Stop texting me", "output": "END"},
            {"input": "I already found a job", "output": "END"},
            {"input": "Please remove me from your list", "output": "END"},
            {"input": "Great, thanks! See you at the interview", "output": "END"},
            {"input": "Thanks for booking the interview", "output": "END"},
            {"input": "I have 5 years of Python experience", "output": "CONTINUE"},
            {"input": "What's the salary range?", "output": "CONTINUE"},
            {"input": "Can we schedule for Tuesday?", "output": "CONTINUE"},
            {"input": "Tell me more about the company", "output": "CONTINUE"},
        ]

    def should_exit(self, user_message: str, conversation_history: str = "") -> bool:
        """
        Determines if the conversation should end.

        Args:
            user_message: Latest message from candidate
            conversation_history: Full conversation context

        Returns:
            True if conversation should end, False otherwise
        """
        decision = self.evaluate(user_message, conversation_history)
        return decision == "END"

    def evaluate(self, user_message: str, conversation_history: str = "") -> str:
        """
        Evaluates the exit decision with full context.

        Returns:
            "END" or "CONTINUE"
        """
        # Build few-shot prompt
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.examples
        )

        # Main prompt with role and instructions
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Exit Detection Advisor for a recruitment chatbot.
Your role is to determine if the conversation should END or CONTINUE.

Rules for END:
- Candidate explicitly says they're not interested
- Candidate asks to stop or be removed
- Candidate says they found another job
- Interview has been successfully confirmed/booked
- Candidate says goodbye after interview is scheduled

Rules for CONTINUE:
- Candidate is asking questions about the job
- Candidate is providing information about their experience
- Candidate is discussing scheduling but hasn't confirmed yet
- Any other ongoing conversation

Output ONLY: END or CONTINUE"""),
            few_shot_prompt,
            ("human", """Conversation History:
{history}

Latest Message: {message}

Decision:""")
        ])

        try:
            chain = final_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "history": conversation_history,
                "message": user_message
            })
            decision = response.strip().upper()
            return "END" if "END" in decision else "CONTINUE"
        except Exception as e:
            print(f"ExitAdvisor error: {e}")
            return "CONTINUE"  # Safe default

    def get_exit_message(self, conversation_history: str, is_interview_booked: bool = False) -> str:
        """
        Generates an appropriate exit message based on conversation outcome.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional Tech Recruiter at TechFlow Solutions.
Generate a brief, polite closing message based on the conversation outcome.

Guidelines:
- If interview was booked: Thank them and confirm they'll receive an invite
- If candidate not interested: Thank them for their time, wish them well
- Keep it professional and concise (1-2 sentences max)
- No emojis"""),
            ("human", """Conversation History:
{history}

Interview Booked: {is_booked}

Generate closing message:""")
        ])

        try:
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({
                "history": conversation_history,
                "is_booked": str(is_interview_booked)
            })
        except Exception as e:
            print(f"Exit message generation error: {e}")
            if is_interview_booked:
                return "Great! Your interview is confirmed. You'll receive a calendar invite shortly. Good luck!"
            return "Thank you for your time. We'll keep your information on file for future opportunities."


def prepare_finetuning_data(conversations_path: str = "sms_conversations.json") -> list:
    """
    Prepares training data for fine-tuning the Exit Advisor.
    Extracts exit-related examples from labeled conversations.

    Returns:
        List of training examples in OpenAI fine-tuning format
    """
    import json

    with open(conversations_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    training_data = []

    for conv in conversations:
        history = ""
        for turn in conv['turns']:
            speaker = "Recruiter" if turn['speaker'] == 'recruiter' else "Candidate"

            # Look for labeled turns to create training examples
            if turn['speaker'] == 'candidate' and turn.get('label'):
                label = turn['label'].upper()
                if label in ['END', 'CONTINUE']:
                    training_data.append({
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an Exit Detection Advisor. Determine if the conversation should END or CONTINUE based on the candidate's message."
                            },
                            {
                                "role": "user",
                                "content": f"History:\n{history}\n\nMessage: {turn['text']}"
                            },
                            {
                                "role": "assistant",
                                "content": label
                            }
                        ]
                    })

            history += f"{speaker}: {turn['text']}\n"

    return training_data


def save_finetuning_data(output_path: str = "exit_advisor_training.jsonl"):
    """Saves fine-tuning data to JSONL file."""
    import json

    data = prepare_finetuning_data()

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved {len(data)} training examples to {output_path}")
    return output_path


if __name__ == "__main__":
    # Test the advisor
    advisor = ExitAdvisor()

    test_cases = [
        "I'm not interested in this position",
        "What's the tech stack like?",
        "Stop messaging me please",
        "Tuesday at 10 works for me",
        "Great, thanks for booking the interview!"
    ]

    print("Testing Exit Advisor:")
    for msg in test_cases:
        result = advisor.evaluate(msg)
        print(f"  '{msg}' -> {result}")
