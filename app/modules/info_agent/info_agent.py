"""
Info Advisor Module
Handles candidate questions using RAG with ChromaDB vector database.
Retrieves relevant information from job description and provides answers.
"""

from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Import embeddings manager from local module
from app.modules.info_agent.ingest import EmbeddingsManager, query_info


class InfoAdvisor:
    """
    Handles candidate questions about the job position.
    Uses RAG (Retrieval Augmented Generation) with ChromaDB for accurate answers.
    Aims to drive conversation toward scheduling an interview.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        self.embeddings_manager = EmbeddingsManager()

        # Few-shot examples for response style
        self.examples = [
            {
                "question": "What's the tech stack?",
                "context": "We use Python with Django/Flask, AWS cloud services, and Docker.",
                "response": "Great question! We primarily use Python with frameworks like Django and Flask for backend development. We're deployed on AWS and use Docker for containerization. Does that align with your experience? I'd love to discuss this more in an interview."
            },
            {
                "question": "What are the requirements?",
                "context": "3+ years Python experience, web frameworks, cloud services knowledge.",
                "response": "We're looking for someone with 3+ years of Python experience, familiarity with web frameworks like Django or Flask, and some cloud services knowledge. Based on our conversation, it sounds like you might be a great fit! Would you like to schedule a call to discuss further?"
            },
            {
                "question": "Is remote work possible?",
                "context": "Location: Tel Aviv (Hybrid)",
                "response": "We offer a hybrid work model based in Tel Aviv - so you'd have flexibility to work from home part of the time. Is that arrangement something that would work for you? If so, I'd love to set up an interview to tell you more."
            }
        ]

    def generate_response(
        self,
        user_message: str,
        conversation_history: str
    ) -> str:
        """
        Generates a response to candidate's question using RAG.

        Args:
            user_message: The candidate's question
            conversation_history: Full conversation context

        Returns:
            Response answering the question and pivoting toward scheduling
        """
        # Retrieve relevant context from vector database
        context = self._get_relevant_context(user_message)

        # Build few-shot prompt
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Question: {question}\nContext: {context}"),
            ("ai", "{response}")
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.examples
        )

        # Main prompt with role, instructions, and few-shot examples
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """# ROLE
You are Alex, the "Conversation Info Advisor" at TechFlow Solutions.
Your job is to retrieve accurate information about the Python Developer job description to answer candidate questions.

# CRITICAL INSTRUCTION: THE HOOK
Your goal is NOT just to answer questions. Your goal is to **drive the conversation toward scheduling an interview**.
Every time you generate an answer:
1. Give the answer clearly and concisely (1-2 sentences max)
2. Append a "Hook" - a transition question that encourages the candidate to prove their skills or agree to an interview

# EXAMPLES OF THE HOOK:
- Q: "What is the tech stack?" -> A: "We primarily use Python 3.10, Django, and PostgreSQL on AWS. **Does your experience align with this stack?**"
- Q: "Is this remote?" -> A: "This is a Hybrid role, requiring 2 days a week in the Tel Aviv office. **Does that work for your schedule?**"
- Q: "What's the salary?" -> A: "The range is competitive, based on experience. **How many years have you been working with Python?**"

# RULES:
1. Use the provided context to answer accurately
2. NEVER just answer - always add the hook question
3. Don't make up information not in the context
4. If you don't know, say so and still add a screening question"""),
            few_shot_prompt,
            ("human", """Conversation History:
{history}

Candidate Question: {question}

Relevant Job Information:
{context}

Generate a helpful response:""")
        ])

        try:
            chain = final_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "history": conversation_history,
                "question": user_message,
                "context": context
            })
            return response.strip()
        except Exception as e:
            print(f"InfoAdvisor error: {e}")
            return self._get_fallback_response(user_message)

    def _get_relevant_context(self, query: str, max_chars: int = 1000) -> str:
        """
        Retrieves relevant context from the vector database.
        """
        try:
            context = self.embeddings_manager.get_relevant_context(query, max_chars=max_chars)
            if context:
                return context
        except Exception as e:
            print(f"Context retrieval error: {e}")

        # Fallback to simple query_info
        return query_info(query)

    def _get_fallback_response(self, question: str) -> str:
        """Provides a fallback response when main generation fails."""
        question_lower = question.lower()

        if any(w in question_lower for w in ['requirement', 'experience', 'need']):
            return "We're looking for 3+ years of Python experience with web frameworks like Django or Flask. Does that match your background? I'd be happy to discuss more details in a quick call."

        elif any(w in question_lower for w in ['salary', 'compensation', 'pay', 'benefit']):
            return "We offer competitive compensation packages. I'd be happy to discuss specifics during an interview. Would you like to schedule a time to chat?"

        elif any(w in question_lower for w in ['stack', 'technology', 'tools']):
            return "We use Python, Django/Flask, AWS, and Docker primarily. Does that align with your experience? I'd love to tell you more in an interview."

        elif any(w in question_lower for w in ['remote', 'location', 'office', 'hybrid']):
            return "We're based in Tel Aviv with a hybrid work model. Is that arrangement something that works for you?"

        else:
            return "That's a great question! I'd be happy to discuss that and more in an interview. What times work best for your schedule?"

    def needs_info_retrieval(self, user_message: str) -> bool:
        """
        Determines if the message requires information retrieval.
        Used by the Main Agent to decide whether to consult this advisor.
        """
        question_indicators = [
            '?', 'what', 'how', 'which', 'when', 'where', 'who', 'why',
            'tell me', 'explain', 'describe', 'requirements', 'salary',
            'benefits', 'stack', 'technology', 'company', 'team',
            'responsibilities', 'experience needed'
        ]

        message_lower = user_message.lower()
        return any(indicator in message_lower for indicator in question_indicators)

    def get_topic_from_question(self, question: str) -> str:
        """
        Identifies the topic of the question for better context retrieval.
        """
        question_lower = question.lower()

        topic_mapping = {
            'requirements': ['requirement', 'experience', 'need', 'qualification', 'must have'],
            'tech_stack': ['stack', 'technology', 'framework', 'tools', 'language'],
            'benefits': ['salary', 'compensation', 'benefit', 'perks', 'pay'],
            'company': ['company', 'about', 'culture', 'team', 'organization'],
            'location': ['remote', 'office', 'location', 'hybrid', 'where'],
            'responsibilities': ['responsibilities', 'do', 'work', 'tasks', 'day to day']
        }

        for topic, keywords in topic_mapping.items():
            if any(kw in question_lower for kw in keywords):
                return topic

        return 'general'


if __name__ == "__main__":
    # Test the advisor
    advisor = InfoAdvisor()

    test_questions = [
        "What's the tech stack?",
        "What are the requirements for this position?",
        "Is remote work possible?",
        "What's the salary range?"
    ]

    history = "Recruiter: Thanks for your interest in our Python Developer position!"

    print("Testing Info Advisor:")
    print("=" * 50)

    for question in test_questions:
        print(f"\nQ: {question}")
        response = advisor.generate_response(question, history)
        print(f"A: {response}")
        print("-" * 40)
