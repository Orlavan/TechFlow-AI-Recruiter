"""
Unit tests for the main application components.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()


class TestMainAgent(unittest.TestCase):
    """Tests for the Main Agent orchestrator."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        if not os.getenv('OPENAI_API_KEY'):
            raise unittest.SkipTest("OPENAI_API_KEY not set")
        
        from app.modules.agents import MainAgent
        cls.agent = MainAgent()
    
    def test_continue_action(self):
        """Test that screening responses return CONTINUE."""
        history = "Recruiter: Tell me about your Python experience.\nCandidate: I have 5 years of experience with Django."
        action = self.agent.decide_action(history)
        self.assertEqual(action, "CONTINUE")
    
    def test_end_action(self):
        """Test that disinterest returns END."""
        history = "Recruiter: Are you interested?\nCandidate: No, please stop contacting me."
        action = self.agent.decide_action(history)
        self.assertEqual(action, "END")


class TestExitAdvisor(unittest.TestCase):
    """Tests for the Exit Advisor."""
    
    @classmethod
    def setUpClass(cls):
        if not os.getenv('OPENAI_API_KEY'):
            raise unittest.SkipTest("OPENAI_API_KEY not set")
        
        from app.modules.exit_agent import ExitAdvisor
        cls.advisor = ExitAdvisor()
    
    def test_should_continue(self):
        """Test that engaged candidate gets CONTINUE."""
        result = self.advisor.should_exit("What's the salary range?")
        self.assertFalse(result)
    
    def test_should_end(self):
        """Test that disinterest gets END."""
        result = self.advisor.should_exit("I'm not interested anymore")
        self.assertTrue(result)


class TestSchedulingAdvisor(unittest.TestCase):
    """Tests for the Scheduling Advisor."""
    
    @classmethod
    def setUpClass(cls):
        if not os.getenv('OPENAI_API_KEY'):
            raise unittest.SkipTest("OPENAI_API_KEY not set")
        
        from app.modules.schedule import SchedulingAdvisor
        cls.advisor = SchedulingAdvisor()
    
    def test_date_parsing(self):
        """Test natural language date parsing."""
        date, time = self.advisor._parse_datetime("Tuesday at 10am")
        self.assertIsNotNone(date)
        self.assertIsNotNone(time)


class TestInfoAdvisor(unittest.TestCase):
    """Tests for the Info Advisor with RAG."""
    
    @classmethod
    def setUpClass(cls):
        if not os.getenv('OPENAI_API_KEY'):
            raise unittest.SkipTest("OPENAI_API_KEY not set")
        
        from app.modules.info_agent import InfoAdvisor
        cls.advisor = InfoAdvisor()
    
    def test_needs_info_detection(self):
        """Test question detection."""
        self.assertTrue(self.advisor.needs_info_retrieval("What's the tech stack?"))
        self.assertTrue(self.advisor.needs_info_retrieval("Tell me about the requirements"))
        self.assertFalse(self.advisor.needs_info_retrieval("Yes I do"))


if __name__ == '__main__':
    unittest.main(verbosity=2)
