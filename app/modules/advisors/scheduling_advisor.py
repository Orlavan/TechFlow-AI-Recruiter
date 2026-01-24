"""
Scheduling Advisor Module
Handles interview scheduling with Function Calling to interact with SQL database.
Parses dates from natural language and manages slot booking.
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Import database tools
from app.modules.database import DatabaseManager, SCHEDULING_TOOLS, execute_function_call


class SchedulingAdvisor:
    """
    Manages interview scheduling using OpenAI Function Calling.
    Interacts with SQL database to check availability and book slots.
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.db = DatabaseManager()

    def handle_scheduling(
        self,
        user_message: str,
        conversation_history: str,
        conversation_start_time: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Handles scheduling requests using Function Calling.

        Args:
            user_message: Latest message from candidate
            conversation_history: Full conversation context
            conversation_start_time: Original conversation timestamp (for date inference)

        Returns:
            Tuple of (response message, is_booking_confirmed)
        """
        # First, classify the scheduling intent
        intent = self._classify_intent(user_message, conversation_history)

        if intent == "REQUEST_AVAILABILITY":
            return self._handle_availability_request(user_message, conversation_history)

        elif intent == "PROPOSE_TIME":
            return self._handle_time_proposal(user_message, conversation_history, conversation_start_time)

        elif intent == "CONFIRM_BOOKING":
            return self._handle_booking_confirmation(user_message, conversation_history)

        else:
            # General scheduling inquiry
            return self._handle_general_inquiry(user_message, conversation_history)

    def _classify_intent(self, user_message: str, history: str) -> str:
        """Classifies the scheduling intent of the message."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Classify the candidate's scheduling intent into one of these categories:

- REQUEST_AVAILABILITY: Asking for available times ("What times work?", "When can we meet?")
- PROPOSE_TIME: Suggesting a specific time ("How about Tuesday at 10?", "Next Friday works")
- CONFIRM_BOOKING: Confirming a proposed time ("Yes, that works", "Let's do it", "Sounds good")
- OTHER: Not about scheduling specifics yet

Output ONLY the category name."""),
            ("human", """Conversation:
{history}

Latest message: {message}

Category:""")
        ])

        try:
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"history": history, "message": user_message})
            return result.strip().upper().replace(" ", "_")
        except Exception as e:
            print(f"Intent classification error: {e}")
            return "OTHER"

    def _handle_availability_request(self, message: str, history: str) -> Tuple[str, bool]:
        """Handles requests for available time slots."""
        # Use Function Calling to get available slots
        messages = [
            {
                "role": "system",
                "content": f"""You are a scheduling assistant. Today's date is {datetime.now().strftime('%Y-%m-%d')}.
The candidate is asking for available interview times. Use the get_available_slots function to retrieve options."""
            },
            {"role": "user", "content": f"Conversation:\n{history}\n\nCandidate: {message}"}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=SCHEDULING_TOOLS,
                tool_choice="auto"
            )

            # Check if function was called
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # Execute the function
                result = execute_function_call(func_name, func_args)
                slots = json.loads(result)

                if slots:
                    slot_list = "\n".join([f"- {s['date']} at {s['time']}" for s in slots])
                    return f"Great! I have the following times available:\n{slot_list}\n\nWhich one works best for you?", False
                else:
                    return "I'm checking our calendar... It seems we're quite busy. Let me have a human recruiter reach out to coordinate.", False

            return "Let me check our availability. What day of the week works best for you?", False

        except Exception as e:
            print(f"Availability request error: {e}")
            slots = self.db.get_available_slots(limit=3)
            if slots:
                slot_list = "\n".join([f"- {s['date']} at {s['time']}" for s in slots])
                return f"Here are some available times:\n{slot_list}\n\nDo any of these work?", False
            return "Could you tell me what days generally work best for your schedule?", False

    def _handle_time_proposal(
        self,
        message: str,
        history: str,
        conversation_start: Optional[str] = None
    ) -> Tuple[str, bool]:
        """Handles when candidate proposes a specific time."""
        # Parse the date/time from the message
        parsed_date, parsed_time = self._parse_datetime(message, conversation_start)

        if parsed_date and parsed_time:
            # Check availability using Function Calling
            messages = [
                {
                    "role": "system",
                    "content": f"Check if the slot {parsed_date} at {parsed_time} is available using check_slot_availability."
                },
                {"role": "user", "content": f"Check availability for {parsed_date} at {parsed_time}"}
            ]

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    tools=SCHEDULING_TOOLS,
                    tool_choice={"type": "function", "function": {"name": "check_slot_availability"}}
                )

                if response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    result = execute_function_call(
                        "check_slot_availability",
                        {"date": parsed_date, "time": parsed_time}
                    )
                    is_available = json.loads(result)

                    if is_available:
                        return f"Let me check... Yes, {parsed_date} at {parsed_time} is available! Shall I book that for you?", False
                    else:
                        # Get alternatives
                        alt_slots = self.db.get_slots_near_date(parsed_date, limit=3)
                        if alt_slots:
                            slot_list = "\n".join([f"- {s['date']} at {s['time']}" for s in alt_slots])
                            return f"Sorry, that slot isn't available. Here are some nearby times:\n{slot_list}\n\nWould any of these work?", False
                        return "That time isn't available. Could you suggest another day or time?", False

            except Exception as e:
                print(f"Time proposal handling error: {e}")

        # Couldn't parse - ask for clarification
        return "Could you specify the exact date and time you're thinking of? For example, 'Tuesday at 10 AM'.", False

    def _handle_booking_confirmation(self, message: str, history: str) -> Tuple[str, bool]:
        """Handles booking confirmation."""
        # Extract the last proposed time from history
        proposed_slot = self._extract_last_proposed_slot(history)

        if proposed_slot:
            date, time = proposed_slot

            # Book using Function Calling
            result = execute_function_call("book_slot", {"date": date, "time": time})
            booking_result = json.loads(result)

            if booking_result.get("success"):
                return f"Excellent! I've confirmed your interview for {date} at {time}. You'll receive a calendar invitation shortly via email.", True
            else:
                return "I apologize, but that slot was just taken. Let me find another time for you.", False

        return "Great! Just to confirm - which time slot would you like me to book?", False

    def _handle_general_inquiry(self, message: str, history: str) -> Tuple[str, bool]:
        """Handles general scheduling inquiries."""
        slots = self.db.get_available_slots(limit=3)

        if slots:
            slot_list = "\n".join([f"- {s['date']} at {s['time']}" for s in slots])
            return f"I'd love to schedule an interview with you. Here are some available times:\n{slot_list}\n\nWhich works best?", False

        return "I'd be happy to help schedule an interview. What days and times generally work for you?", False

    def _parse_datetime(
        self,
        text: str,
        reference_date: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses date and time from natural language.
        Uses conversation start time as reference for relative dates.
        """
        text = text.lower()

        # Reference date for relative calculations
        if reference_date:
            try:
                ref = datetime.fromisoformat(reference_date.replace('Z', '+00:00'))
            except:
                ref = datetime.now()
        else:
            ref = datetime.now()

        # Day name mapping
        day_names = {
            'sunday': 6, 'monday': 0, 'tuesday': 1, 'wednesday': 2,
            'thursday': 3, 'friday': 4, 'saturday': 5
        }

        target_date = None

        # Check for "tomorrow"
        if 'tomorrow' in text:
            target_date = ref + timedelta(days=1)

        # Check for "next [day]"
        elif 'next' in text:
            for day_name, weekday in day_names.items():
                if day_name in text:
                    days_ahead = weekday - ref.weekday()
                    if days_ahead <= 0:
                        days_ahead += 7
                    days_ahead += 7  # "next" means next week
                    target_date = ref + timedelta(days=days_ahead)
                    break

        # Check for day names without "next"
        else:
            for day_name, weekday in day_names.items():
                if day_name in text:
                    days_ahead = weekday - ref.weekday()
                    if days_ahead <= 0:
                        days_ahead += 7
                    target_date = ref + timedelta(days=days_ahead)
                    break

        # Parse time
        time_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?'
        time_match = re.search(time_pattern, text)

        target_time = None
        if time_match:
            hour = int(time_match.group(1))
            minute = time_match.group(2) or '00'
            period = time_match.group(3)

            if period and period.lower() == 'pm' and hour < 12:
                hour += 12
            elif period and period.lower() == 'am' and hour == 12:
                hour = 0

            target_time = f"{hour:02d}:{minute}"

        date_str = target_date.strftime('%Y-%m-%d') if target_date else None
        return date_str, target_time

    def _extract_last_proposed_slot(self, history: str) -> Optional[Tuple[str, str]]:
        """Extracts the last proposed time slot from conversation history."""
        # Look for date patterns in recent recruiter messages
        date_pattern = r'(\d{4}-\d{2}-\d{2})\s+at\s+(\d{2}:\d{2})'
        matches = re.findall(date_pattern, history)

        if matches:
            return matches[-1]  # Return most recent
        return None


if __name__ == "__main__":
    # Test the advisor
    advisor = SchedulingAdvisor()

    test_messages = [
        "What times are available?",
        "How about Tuesday at 10am?",
        "Yes, that works for me!"
    ]

    history = "Recruiter: Would you like to schedule an interview?"

    for msg in test_messages:
        response, booked = advisor.handle_scheduling(msg, history)
        print(f"Input: {msg}")
        print(f"Response: {response}")
        print(f"Booked: {booked}")
        print("-" * 40)
        history += f"\nCandidate: {msg}\nRecruiter: {response}"
