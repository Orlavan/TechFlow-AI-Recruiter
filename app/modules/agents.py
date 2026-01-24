"""
TechFlow AI Recruiter - Main Agents Module

Contains the Main Agent (Orchestrator) and RecruitmentBot manager.
Uses LangChain with advanced prompting strategies:
- Role-based system prompts
- Few-shot learning examples
- API parameter optimization
"""

import os
import re
from datetime import datetime
from typing import Tuple, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Import Advisor Agents
from app.modules.exit_agent.exit_agent import ExitAdvisor
from app.modules.schedule.scheduling_agent import SchedulingAdvisor
from app.modules.info_agent.info_agent import InfoAdvisor

# Model Configuration
MAIN_MODEL = os.getenv("MAIN_MODEL", "gpt-4")
FINETUNED_MAIN_MODEL = os.getenv("FINETUNED_MAIN_MODEL", None)


# =============================================================================
# MAIN AGENT (ORCHESTRATOR)
# =============================================================================

class MainAgent:
    """
    Main Orchestrator Agent - The 'Brain' of the system.

    Analyzes conversation context and decides the next action:
    - CONTINUE: Keep screening the candidate
    - SCHEDULE: Move to interview scheduling
    - END: Conclude the conversation
    """

    def __init__(self, use_finetuned: bool = True):
        """Initialize the Main Agent with LLM and few-shot examples."""

        model = FINETUNED_MAIN_MODEL if (use_finetuned and FINETUNED_MAIN_MODEL) else MAIN_MODEL
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,   # Deterministic routing
            max_tokens=10    # Only need one word
        )

        # Few-shot examples for decision making
        self.examples = [
            # CONTINUE examples
            {"history": "Recruiter: Tell me about your Python experience.\nCandidate: I have 5 years of experience with Django.",
             "action": "CONTINUE"},
            {"history": "Recruiter: Great experience!\nCandidate: What's your tech stack?",
             "action": "CONTINUE"},
            {"history": "Recruiter: We use Python and AWS.\nCandidate: That sounds interesting. Can you tell me about the team?",
             "action": "CONTINUE"},

            # SCHEDULE examples
            {"history": "Recruiter: Would you like to schedule an interview?\nCandidate: Yes, I'd look forward to that.",
             "action": "SCHEDULE"},
            {"history": "Recruiter: Here are available times: Tuesday 10am, Wednesday 2pm\nCandidate: Tuesday at 10 works for me.",
             "action": "SCHEDULE"},
            {"history": "Recruiter: That time isn't available.\nCandidate: What about Thursday then?",
             "action": "SCHEDULE"},
            {"history": "Recruiter: Let's set up a call.\nCandidate: Sure, when are you free?",
             "action": "SCHEDULE"},

            # END examples
            {"history": "Recruiter: Would you like to hear more?\nCandidate: No thanks, I'm not interested.",
             "action": "END"},
            {"history": "Recruiter: Can we schedule a call?\nCandidate: Please stop texting me.",
             "action": "END"},
            {"history": "Recruiter: Your interview is confirmed for Tuesday at 10am.\nCandidate: Great, thanks!",
             "action": "END"},
            {"history": "Recruiter: Looking forward to meeting you!\nCandidate: Thanks, see you then!",
             "action": "END"},
        ]

    def decide_action(self, conversation_history: str) -> str:
        """
        Analyze conversation and decide the next action.

        Args:
            conversation_history: Full conversation context

        Returns:
            Action: "CONTINUE", "SCHEDULE", or "END"
        """
        # Build few-shot prompt
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Conversation:\n{history}\n\nAction:"),
            ("ai", "{action}")
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.examples
        )

        # System prompt with role and rules
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """# ROLE
You are a Technical Recruiter for TechFlow Solutions interviewing candidates for a Senior Python Developer position.

# OBJECTIVE
1. Verify the candidate's qualifications
2. If qualified: Schedule an interview (SCHEDULE)
3. If unqualified or uninterested: End politely (END)

# RULES
1. Lead the conversation - don't let candidates go off-topic
2. Answer questions, but pivot back to screening
3. No SCHEDULE until 2+ screening details verified
4. If candidate is rude or uninterested: END

# DECISION LOGIC
- CONTINUE: Need more screening, answering questions
- SCHEDULE: Candidate passed screening AND wants to proceed
- END: Not interested, rude, or interview confirmed

Output ONLY: CONTINUE, SCHEDULE, or END"""),
            few_shot_prompt,
            ("human", "Conversation:\n{history}\n\nAction:")
        ])

        try:
            chain = final_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"history": conversation_history})
            action = response.strip().upper()

            # Validate and return
            if action in ["CONTINUE", "SCHEDULE", "END"]:
                return action
            if "CONTINUE" in action: return "CONTINUE"
            if "SCHEDULE" in action: return "SCHEDULE"
            if "END" in action: return "END"
            return "CONTINUE"  # Default fallback

        except Exception as e:
            return f"ERROR: {str(e)}"


# =============================================================================
# RECRUITMENT BOT (MANAGER)
# =============================================================================

class RecruitmentBot:
    """
    Main Manager Class - Orchestrates the recruitment conversation.

    Architecture:
    - Main Agent: Decides action (CONTINUE/SCHEDULE/END)
    - Exit Advisor: Validates END decisions (fine-tuned)
    - Scheduling Advisor: Handles interview booking (Function Calling)
    - Info Advisor: Answers questions (RAG)
    """

    def __init__(self):
        """Initialize the bot with all advisor components."""

        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in .env file")

        # Initialize agents
        self.main_agent = MainAgent()
        self.exit_advisor = ExitAdvisor()
        self.scheduling_advisor = SchedulingAdvisor()
        self.info_advisor = InfoAdvisor()

        # State tracking
        self.is_interview_booked = False
        self.conversation_start = datetime.now().isoformat()
        self.screening_details_collected = 0
        self.screening_complete = False

    def process_turn(
        self,
        user_input: str,
        history: str,
        conversation_timestamp: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Process a single conversation turn.

        Flow:
        1. Update screening progress
        2. Main Agent decides action
        3. Validation layer (Exit Advisor confirms END)
        4. Route to appropriate advisor

        Args:
            user_input: Candidate's message
            history: Conversation history
            conversation_timestamp: Start time for date inference

        Returns:
            Tuple of (response, action)
        """
        current_history = history + f"\nCandidate: {user_input}"

        # Update screening progress
        self._update_screening_progress(user_input)

        # Main Agent decides action
        action = self.main_agent.decide_action(current_history)

        # === VALIDATION LAYER ===

        # Validate END decisions
        if action == "END":
            if not self.exit_advisor.should_exit(user_input, current_history):
                if not self.is_interview_booked:
                    action = "CONTINUE"

        # Check for disengagement
        if action == "CONTINUE" and self._is_disengaged(user_input, current_history):
            if self.exit_advisor.should_exit(user_input, current_history):
                action = "END"

        # Block SCHEDULE if screening incomplete
        if action == "SCHEDULE" and not self.screening_complete:
            action = "CONTINUE"

        # === RESPONSE GENERATION ===

        if action == "CONTINUE":
            if self.info_advisor.needs_info_retrieval(user_input):
                response = self.info_advisor.generate_response(user_input, current_history)
            else:
                response = self._generate_screening_response(user_input, current_history)

        elif action == "SCHEDULE":
            response, self.is_interview_booked = self.scheduling_advisor.handle_scheduling(
                user_input, current_history, conversation_timestamp or self.conversation_start
            )
            if self.is_interview_booked:
                action = "END"

        elif action == "END":
            response = self.exit_advisor.get_exit_message(
                current_history, is_interview_booked=self.is_interview_booked
            )

        else:
            response = "I'm sorry, I encountered an error. Please try again."

        return response, action

    def _update_screening_progress(self, user_input: str) -> None:
        """Track screening progress based on candidate responses."""
        user_lower = user_input.lower()

        # Check for experience mentioned
        exp_keywords = ['year', 'experience', 'worked for', 'been working']
        if any(kw in user_lower for kw in exp_keywords):
            if re.search(r'\d+', user_input):
                self.screening_details_collected += 1

        # Check for tech stack
        tech_keywords = ['python', 'django', 'flask', 'aws', 'docker', 'sql',
                        'fastapi', 'react', 'kubernetes', 'cloud']
        if any(kw in user_lower for kw in tech_keywords):
            self.screening_details_collected += 1

        # Complete if 2+ details
        if self.screening_details_collected >= 2:
            self.screening_complete = True

    def _is_disengaged(self, user_input: str, history: str) -> bool:
        """Detect if candidate is disengaged or refusing to participate."""
        user_lower = user_input.lower().strip()

        disengaged = ['nope', 'no', 'nah', 'not interested', 'stop',
                     'leave me alone', 'go away', "don't want"]

        if any(phrase in user_lower for phrase in disengaged):
            neg_count = (history.lower().count('candidate: no') +
                        history.lower().count('candidate: nope'))
            if neg_count >= 2:
                return True

        refusals = ["don't want to tell", "won't tell", 'not interested', 'leave me alone']
        return any(phrase in user_lower for phrase in refusals)

    def _generate_screening_response(self, user_input: str, history: str) -> str:
        """Generate a screening response to keep conversation moving."""
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Alex, a Tech Recruiter at TechFlow Solutions.
You're screening for a Python Developer position.

RULES:
1. Always end with a screening question
2. Keep responses brief (1-2 sentences + question)
3. Ask about: Python experience, Django/Flask, databases, cloud (AWS/GCP)
4. If vague answer: ask to elaborate"""),
            ("human", "Conversation:\n{history}\n\nCandidate: {input}\n\nRespond:")
        ])

        try:
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"history": history, "input": user_input})
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def reset(self):
        """Reset bot state for new conversation."""
        self.is_interview_booked = False
        self.conversation_start = datetime.now().isoformat()
        self.screening_details_collected = 0
        self.screening_complete = False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_bot() -> RecruitmentBot:
    """Factory function to create a new RecruitmentBot."""
    return RecruitmentBot()


if __name__ == "__main__":
    # Quick test
    agent = MainAgent()

    test_cases = [
        "Recruiter: Tell me about your experience.\nCandidate: I have 5 years with Python.",
        "Recruiter: Would you like to schedule?\nCandidate: Yes, that sounds great!",
        "Recruiter: Tell me more.\nCandidate: I'm not interested, please stop."
    ]

    print("Testing Main Agent:")
    print("=" * 50)
    for history in test_cases:
        action = agent.decide_action(history)
        print(f"Action: {action}")
