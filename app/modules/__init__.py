"""
Application Modules Package
Contains all AI agents and advisors for the recruitment bot.
"""

# Re-export from new module structure for backwards compatibility
from app.modules.agents import MainAgent, RecruitmentBot, create_bot
from app.modules.exit_agent import ExitAdvisor
from app.modules.info_agent import InfoAdvisor
from app.modules.schedule import SchedulingAdvisor

__all__ = [
    "MainAgent",
    "RecruitmentBot", 
    "create_bot",
    "ExitAdvisor",
    "InfoAdvisor",
    "SchedulingAdvisor"
]
